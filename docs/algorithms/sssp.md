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
self-loops included), and the tree is brought up to date. Postcondition: the result equals
`compute()` on the new graph, bit for bit.

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
| `sssp.loop` | propagate decreases until no distance changes (near-far worklist on OpenMP; re-scan rounds on the sequential backend) |
| `sssp.finalize` | unpack distances and parents; count `affected` |

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
| sequential | hook-by-hook reference (`engine::operators`) | `sequentialSOSPUpdate` adapted |
| openmp | the ported paper engine (`engine::fused`) | `sospUpdateCpu` / `sospFromScratchCpu` ported straight |
| cuda | M1b | `sospUpdateGpu` (persistent cooperative kernel) |

All backends return identical trees; `invalidated` and `affected` are deterministic,
`iterations`, `epochs` and `pushes` depend on the schedule.

## 5. Performance notes

The paper-timed region of the original, `obj<k>/sosp_update_compute` (sospUpdateCpu), maps to
`sssp.identify_affected + sssp.seed + sssp.loop + sssp.finalize` (`parity/timed_regions/sssp.toml`).
The OpenMP engine allocates nothing after the result is created (the workspace is reserved and
touched at compute / from_arrays). Each result owns a workspace of about 38 bytes per vertex.

## 6. Limitations

- Integer weights of at least 1 only (zero or negative weights are rejected).
- The graph must store its in-edges (`graph_properties::store_transposed`, the default).
- Distances must fit 62 bits ((n - 1) * max weight); beyond that `compute` and `update` throw.
- A tree imported with `validate_inputs = false` must be a shortest-path tree of the graph; the
  OpenMP engine still rejects a parent cycle (the result is then poisoned), the sequential engine
  does not detect every corrupt input.

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
| `SospWorkspace` | the workspace inside `sssp::result` (reserved once) |
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
