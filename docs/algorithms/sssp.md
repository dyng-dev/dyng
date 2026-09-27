# sssp: dynamic single-source shortest paths

Maturity: **experimental** (M1a: sequential and OpenMP backends; CUDA arrives in M1b, the Python
binding in M5). Header: `<dyng/sssp.hpp>`. Oracle: `compute`. Determinism: `bitwise`.

## 1. Problem

Input: a directed (or symmetric) graph with K integer weight columns (weights in
[1, 2^31 - 1]), a source vertex and an objective column. Output: the distance of every vertex
from the source (`infinite_distance<int64_t>()` if unreachable) and the **canonical**
shortest-path tree: the parent of v is the lowest-id in-neighbour u with
dist[u] + w(u,v) == dist[v]; -1 for the source and unreachable vertices.

Update model: a batch of insertions (upserts: an insertion of an existing edge changes its
weights), deletions and weight changes is applied to the graph under its `batch_semantics`
(`graph_properties::mosp_compatible()` reproduces MOSP's `applyChangeBatch()`, parallel edges and
self-loops included), and the tree is brought up to date. Postcondition: the distances equal
`compute()`'s on the new graph and the tree is a shortest-path tree of it; from a canonical input
tree (every result of `compute()`, or `from_arrays(..., canonicalize = true)`) the result equals
`compute()` bit for bit.

**Tie rule** (the rule of MOSP-OpenMP's `sospUpdateCpu` and MOSP-CUDA's `sospUpdateGpu`, on every
backend): a parent changes only when its vertex is re-evaluated. The vertices of the invalidated
subtrees and the insertion heads take their best (distance, lowest parent id) over all
in-neighbours; the out-neighbours x of a vertex a whose distance decreased adopt
(d(a) + w(a,x), a) if that pair is smaller than their own; all other vertices keep their parents.
So a tree adopted with `canonicalize = false` whose tie parents are not the lowest ids (MOSP's
dataset trees are such trees) keeps those parents where the batch does not reach, exactly as the
original `mosp` driver does; the distances are still those of `compute()`. In the distance-only
mode (`stats::packed_parents == false`) every parent is recovered with the lowest-id rule after
the search, so the result equals `compute()` on every input. MOSP-OpenMP's own sequential update
(`sequentialSOSPUpdate`) re-scans all in-neighbours of every candidate and can pick a lower-id
tight parent from a vertex that did not improve; dynG's sequential backend follows the rule
above instead (ADR 0006), so that every backend returns the same tree.

## 2. Template mapping

```
normalize -> translate -> prepare -> [before_apply] -> commit ->
identify_affected -> seed -> { FP: loop until is_converged } -> finalize
```

| Hook (profiler stage) | sssp |
|---|---|
| `sssp.prepare` (G_t) | largest weight (old graph and batch) and weight sum of the objective; the default near-far width; batch weights checked |
| `sssp.commit` | `graph::apply` with the per-edge classification (`apply_delta`): deletions and per-objective weight increases |
| `sssp.identify_affected` | roots = heads v of deleted or weight-increased edges (u,v) with parent[v] == u; their subtrees are invalidated |
| `sssp.seed` | invalidated vertices and insertion heads pull their best (distance, lowest id) over their in-neighbours |
| `sssp.loop` | propagate decreases until no distance changes (near-far worklist on OpenMP; rounds of the same push rule on the sequential backend) |
| `sssp.finalize` | unpack distances and parents (the distance-only mode recovers every parent); count `affected` |

`compute()` is the static enactor: `sssp.reset` -> `sssp.seed` (the source) -> `sssp.loop` ->
`sssp.finalize`, inside `sssp.compute`. `update()` runs inside `sssp.update`.

The four Chapter 3 challenges: (i) affected set: subtree invalidation below changed tree edges
(this replaces the thesis' "best current in-neighbour" rule, which counts to infinity); (ii)
propagation scope: only vertices whose distance decreased are expanded; distances only
decrease, so there is no iteration cap; (iii) correctness under parallelism: an atomic minimum on
a packed (distance, parent) word gives canonical ties without locks; (iv) data structure: the
compact CSR with stored in-edges and objective-major weight columns.

## 3. API

```cpp
auto res = dyng::resources::openmp(28);              // or resources::sequential()
auto g = dyng::graph<std::int32_t, std::int32_t, std::int32_t>::from_csr(
    res, csr.view(), dyng::graph_properties::mosp_compatible());
auto tree = dyng::sssp::compute(res, g, /*source=*/0);
dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);
// several objectives on one graph: the batch is applied once
auto [s0, s1] = dyng::update(res, g, batch.view(), tree0, tree1);
```

| Option | Default | Meaning |
|---|---|---|
| `delta` | 0 (automatic) | near-far bucket width: max(1, 32 * average weight / average out-degree) of the graph before the batch |
| `objective` | 0 | the weight column (fixed at compute) |
| `cuda_engine` | automatic | CUDA engine (M1b); ignored on the CPU |
| `validate_inputs` | true | O(n) checks of trees adopted with `result::from_arrays()` |

Python: planned (M5).

## 4. Backends and determinism

| Backend | Engine | Origin |
|---|---|---|
| sequential | hook-by-hook reference (`engine::operators`) | `sequentialSOSPUpdate` adapted (Step 2 with `sospUpdateCpu`'s push rule) |
| openmp | the ported paper engine (`engine::fused`) | `sospUpdateCpu` / `sospFromScratchCpu` ported straight |
| cuda | M1b | `sospUpdateGpu` (persistent cooperative kernel) |

All backends return identical trees, from canonical and non-canonical input trees alike (the
randomized test `NonCanonicalInputTreesAgreeOnEveryBackend` perturbs tie parents); `invalidated`
and `affected` are deterministic, `iterations`, `epochs` and `pushes` depend on the schedule.

## 5. Performance notes

The paper-timed region of the original, `obj<k>/sosp_update_compute` (sospUpdateCpu), maps to
`sssp.identify_affected + sssp.seed + sssp.loop + sssp.finalize` (`parity/timed_regions/sssp.toml`).

**Scratch memory (ADR 0015).** The engines' workspace (about 38 bytes per vertex: packed words,
stamps, flags and six frontier lists) belongs to the `resources` handle, not to the result.
`compute()`, `from_arrays()`, `clone()` and each result's part of `update()` lease it from the
handle and size it once (profiler stage `sssp.workspace`), so results run through one handle
share it: the K objectives of `dyng::update_each()` use one workspace one after the other, as
MOSP's `mospUpdate()` shares its `SospWorkspace`, and a steady-state update allocates no scratch
memory. Calls that run concurrently on copies of one handle lease distinct workspaces.
`resources::workspace_bytes()` reports the cached bytes, `resources::release_workspaces()` frees
them. As in the original, objective 0 of an update first touches the frontier-list pages it uses.

**The graph around it.** A graph built with `from_csr()` / `from_edges()` does not transpose
itself: the in-edges are built on first use, which for an update is inside the commit, for the
updated graph only (MOSP builds its reverse graph once, in "prepare"). `from_csr()` takes a CSR
that already has the requested form (any CSR under `mosp_compatible()`) as it is, and its rvalue
overload takes over the arrays without a copy. On the OpenMP backend the batch's CSR assembly,
the checks of `from_csr()` and the validation of imported trees (`validate_inputs`) run in
parallel, with results and messages identical to the sequential ones.

**Measured against the originals** (OpenMP, 28 threads pinned, medians of 21 alternating runs,
`parity/results/M1b.md`): every objective's SOSP region, the apply region and the end-to-end time
of `dyng-compat-mosp` against MOSP-OpenMP@c352151's `mosp` on roadNet-PA, roadNet-CA,
rgg_n_2_20_s0 and road_usa (50K safe, 50K unsafe, 10K local batches). Ratio dynG / original
(medians; below 1 is faster), the range over the three batches and the objectives:

| Graph | SOSP region per objective (gate 1.05x) | apply (gate 1.10x) | end to end (gate 1.10x) |
|---|---|---|---|
| roadNet-PA | 0.78-0.95x | 0.78-0.91x | 0.80-0.83x |
| roadNet-CA | 0.81-0.99x | 0.84-0.88x | 0.76x |
| rgg_n_2_20_s0 | 0.82-1.01x | 0.77-0.80x | 0.83-0.85x |
| road_usa | 0.84-0.96x | 0.79x | 0.77-0.81x |

Byte-identical outputs and equal `invalidated` counters in every run; details in the record.

## 6. Limitations

- Integer weights of at least 1 only (zero or negative weights are rejected).
- The graph must store its in-edges (`graph_properties::store_transposed`, the default).
- Distances must fit 62 bits ((n - 1) * max weight); beyond that `compute` and `update` throw.
- A tree imported with `validate_inputs = false` must be a shortest-path tree of the graph. Both
  engines still reject a parent cycle that no root of the batch breaks (the same check and
  message; the batch is already applied, so the result is then poisoned and every later use of it
  throws `stale_result_error` until it is recomputed); other corrupt inputs (for example wrong
  distances) give undefined, though memory-safe, results.

## 7. Differences from the paper

The code follows the corrected MOSP-OpenMP@c352151, not the thesis pseudocode: subtree
invalidation plus a pull pass replaces Step 1's best-in-neighbour rule (which counts to infinity,
e.g. d(1) = 60 instead of 90 on the n = 6 stress seed), the propagation is a monotone near-far
worklist without an iteration cap or a reachability pass, and ties go to the lowest parent id.

## 8. Mapping from the original code

| MOSP-OpenMP@c352151 | dynG |
|---|---|
| `sospUpdateCpu` | `sssp::update` on `resources::openmp` (`cpp/src/algorithms/sssp/openmp.cpp`) |
| `sospFromScratchCpu`, `mospPrep init` / `dijkstraCsrGraph` | `sssp::compute`; `testing::dijkstra` |
| `sequentialSOSPUpdate` | `sssp::update` on `resources::sequential` (`sequential.cpp`) |
| `SospWorkspace` (one per `mospUpdate()` call, shared by the objectives) | `detail::sssp_workspace`, leased from the pool of `resources` (one per handle, shared by the results; ADR 0015) |
| `SospStats` | `sssp::stats` (`invalidated`, `iterations`, `epochs`, `pushes`, `packed_parents`) |
| `HostChanges` (changed edges, insert heads), `weightIncreaseMask` | `detail::apply_delta` from the commit |
| `defaultDelta` | `options.delta = 0` |
| `canonicalizeTree`, `--init` files | `sssp::result::from_arrays(..., canonicalize)` |
| `checkSospTree` | `testing::check_sssp_tree` |
| `ListGather` | `detail::list_gather` (`cpp/src/util/list_gather.hpp`) |
| `mosp` driver (per-objective part) | `tools/compat` `dyng-compat-mosp` |

## 9. How to cite

`dyng::citation("sssp")`: DynaMOSP (IPDPS 2025) and its journal version (IEEE TPDS 2025), keys
`dynamosp2025` and `dynamosptpds2025` in `docs/references.bib`.
