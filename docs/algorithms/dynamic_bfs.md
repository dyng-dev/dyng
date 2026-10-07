# dynamic_bfs: BFS levels under edge batches (tutorial)

Maturity: **tutorial** (teaching material, not a research algorithm). Header:
`<dyng/dynamic_bfs.hpp>`. Python: `dyng.dynamic_bfs`. Oracle: `compute`. Determinism:
`exact_value`. Backends: sequential, openmp, cuda. Family: fixed point, Tier A (the framework's
operators).

:::{note}
`dynamic_bfs` is one of the two **teaching algorithms** of dynG (PLAN Section 9.3). It is small
enough to read in one sitting, exercises the whole fixed-point template (classify, invalidate a
subtree, seed, advance, convergence) and runs on every backend from one source. It is complete
and checked by the conformance kit like every algorithm, but it is not tuned for speed and comes
from no paper. The tutorial {doc}`../tutorials/your_first_dynamic_algorithm` builds a simpler
version step by step.
:::

## 1. Problem

Input: a directed graph (weights are ignored) with its in-edges stored
(`graph_properties::store_transposed`, the default) and a source vertex (`options::source`, fixed at
`compute()`). Output: the BFS level of every vertex, the number of edges on a shortest path from
the source (`result::levels()`, int64), -1 for the vertices the source does not reach.

The update model: a batch of edge insertions and deletions (vertex growth allowed: new vertices
start unreached). An insertion `u -> v` can only lower levels, a deletion can only raise them, and
only below the deleted edge. `update()` applies the batch and repairs the levels; afterwards they
equal `compute()` on the new graph exactly.

The result also keeps a BFS tree (one parent per vertex, internal) to know which vertices a
deletion can affect.

## 2. Template mapping

```
commit -> identify_affected -> seed -> { loop until the frontier is empty } -> finalize
```

`detail::dynamic_bfs_problem` (`cpp/src/algorithms/dynamic_bfs/problem.hpp`, the template card at
its top) says what happens when; `dynamic_bfs_engine_impl<exec_t>`
(`cpp/src/algorithms/dynamic_bfs/engine.hpp`) says how, once for every backend: each pass is a
`DYNG_HD` functor run by an executor of the framework operators (`cpp/src/operators/`), a loop
(`sequential.cpp`), an OpenMP loop (`openmp.cpp`) or a CUDA kernel (`cuda.cu`).

| Step | Hook (profiler stage) | What it does | Operator (PLAN 4.5.3) |
|---|---|---|---|
| apply | commit (`dynamic_bfs.commit`) | the batch applied once; the in-edges (host) or the device copy with its in-edges (cuda) prepared for the engines | – |
| 1b | `identify_affected` (`dynamic_bfs.identify_affected`) | a deleted edge `u -> v` with `parent[v] == u` and no parallel `u -> v` left makes `v` a root; the roots and their tree descendants are invalidated (level and parent -1), one pass per tree level | `classify`, `invalidate_subtree` (a level-by-level walk) |
| 1b | `seed` (`dynamic_bfs.seed`) | every invalidated vertex pulls the best level of its valid in-neighbours; every inserted edge `u -> v` that is in G_{t+1} offers `level[u] + 1` to `v` (an atomic minimum; the batch's insertion list holds the requested insertions, and with `deletions_first = false` a batch may insert an edge and delete it again); a vertex whose level dropped enters the frontier | `neighbor_reduce` (pull), `advance` |
| 2 | `loop` (`dynamic_bfs.loop`, one call per round) | every frontier vertex offers `level + 1` to its out-neighbours; the enactor swaps the frontiers and stops when the frontier is empty (the default `is_converged`) | `advance` (push) with a frontier push deduplicated by round stamps |
| finish | `finalize` (`dynamic_bfs.finalize`) | the parent of every touched vertex becomes its lowest-id in-neighbour one level up; the invalidation flags are cleared; the stats | – |

`compute()` is the static enactor: `dynamic_bfs.reset` -> `dynamic_bfs.seed` (the source at level
0) -> `dynamic_bfs.loop` -> `dynamic_bfs.finalize`, the same passes from scratch.

The four Chapter 3 challenges: (i) **affected set**: the subtrees under the deleted tree edges
(their levels may grow) and the heads of the inserted edges (their levels may drop); a vertex
outside the subtrees keeps its tree path, so its level is still an achievable upper bound;
(ii) **propagation scope**: a frontier of vertices whose level dropped, round by round; (iii)
**correctness under parallelism**: offers are atomic minima on the level word (-1, unreached, is
the largest unsigned value), so the fixed point does not depend on the order of the offers, and
the parents are recomputed from the final levels by a deterministic rule, so every backend keeps
the same tree; (iv) **data structure**: the compact CSR with its in-edges (the pull and the parent
repair read them).

## 3. API

```cpp
#include <dyng/dynamic_bfs.hpp>

dyng::dynamic_bfs::options opt;
opt.source = 0;
auto bfs = dyng::dynamic_bfs::compute(res, g, opt);
dyng::dynamic_bfs::stats s = dyng::dynamic_bfs::update(res, g, batch.view(), bfs);
// bfs.levels(): one int64 per vertex, -1 if unreachable (device memory for a cuda result)
```

```python
import dyng
g = dyng.Graph.from_edges([0, 0, 1, 2, 3], [1, 2, 3, 3, 4])
bfs = dyng.dynamic_bfs.compute(g, 0)
stats = dyng.dynamic_bfs.update(g, dyng.EdgeBatch(delete=([1], [3])), bfs)
print(bfs.levels.to_numpy(), stats.invalidated)   # [0 1 1 2 3] 2
```

| Option | Default | Meaning |
|---|---|---|
| `source` | 0 | the source vertex; fixed at `compute()` |

| Stat | Deterministic | Meaning |
|---|---|---|
| `affected` | yes | vertices whose level changed |
| `invalidated` | yes | vertices invalidated (the subtrees under the deleted tree edges) |
| `invalidation_rounds` | yes | passes of the invalidation: the roots, then one per tree level (0 without deletions) |
| `iterations` | no | loop rounds |
| `frontier_visits` | no | frontier vertices expanded |
| `batch` | yes | what applying the batch did to the graph |

Graph types: `graph<int32_t, int32_t or int64_t, int32_t or unweighted>` and
`graph<int64_t, int64_t, int32_t>`. `dyng::update(res, g, batch, r1, r2, ...)` accepts
`dynamic_bfs` results in C++; the Python `dyng.update()` and the CLI do not (ADR 0030).

## 4. Backends, engines and determinism

One engine (Tier A, `stats.engine_used == engine::operators`) on all three backends; the levels
and the deterministic counters are identical on every backend and run (conformance checks C3 and
C6). On cuda the result lives on the device, and the host synchronizes once per count it reads:
one per invalidation pass, one for the seed, one per loop round and two in finalize
(`invalidation_rounds + iterations + 3`, the budget that conformance check C8 holds it to).

## 5. Complexity and performance notes

An update costs the out-edges of the invalidated subtrees, the in-edges of the invalidated and
touched vertices, and the out-edges of every frontier vertex per round it is on the frontier,
plus an O(n) copy of the levels for `stats.affected` (a production algorithm would track the
touched vertices' old levels instead). The profiler stages are the hooks of the table above. There
is no benchmark suite and no performance gate: `dynamic_bfs` is teaching material.

## 6. Limitations

- Not tuned: one kernel launch per pass and one host synchronization per round on cuda; no
  direction-optimizing (pull) rounds, no bucketed frontier.
- The O(n) snapshot of the levels per update (see 5).
- Unweighted hops only: for weighted shortest paths use {doc}`sssp`.

## 7. Differences from the paper

Not from a paper: a teaching algorithm written for dynG (PLAN Section 9.3).

## 8. Mapping from the original code

None (not a port).

## 9. How to cite

`dyng::citation("dynamic_bfs")` returns the library's BibTeX entry (the manifest cites no paper).
