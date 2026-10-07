# triangle_delta: the triangle count under edge batches (tutorial)

Maturity: **tutorial** (teaching material, not a research algorithm). Header:
`<dyng/triangle_delta.hpp>`. Python: `dyng.triangle_delta`. Oracle: `compute`. Determinism:
`exact_value`. Backends: sequential, openmp, cuda. Family: aggregate delta, Tier A (the
framework's operators), ownership `min_member`.

:::{note}
`triangle_delta` is one of the two **teaching algorithms** of dynG (PLAN Section 9.3): the
aggregate-delta companion of {doc}`dynamic_bfs`. It exercises the other half of the template: a
count on the old graph before the commit, a count on the new graph after it, and an ownership
rule that counts every pattern exactly once. Triangle counting is a special case of the pattern
counts of `triad_count` (ESCHER), so this page also prepares the "count a new hypergraph pattern"
tutorial. It is complete and checked by the conformance kit, but not tuned and from no paper.
:::

## 1. Problem

Input: an **undirected** simple graph (`graph_properties::directed = false`, sorted rows, no
parallel edges; self-loops are ignored, and so are weights). Output: the number of triangles
{a, b, c} (`result::count()`, uint64, host memory on every backend).

The update model: a batch of edge insertions and deletions; an undirected graph changes both
directions of every batch edge, and new vertices may appear. A batch changes the count by what it
destroys and creates:

```text
T(G_{t+1}) = T(G_t) - #{triangles of G_t through a deleted edge}
                    + #{triangles of G_{t+1} through an inserted edge}
```

(a triangle of G_t without a deleted edge is a triangle of G_{t+1} without an inserted one, and
the other way round). Afterwards the count equals `compute()` on the new graph exactly.

### Worked example

K4 without the edge {2, 3} has two triangles, {0, 1, 2} and {0, 1, 3}. Inserting {2, 3} creates
two (the common neighbours 0 and 1 of 2 and 3): K4 has four. Deleting {0, 1} and {0, 2} then
destroys three triangles; {0, 1, 2} goes through both deleted edges and is counted once, by the
smaller edge {0, 1}. One triangle is left, {1, 2, 3}. (`cpp/tests/algorithms/triangle_delta/
triangle_delta_test.cpp`, `ThePageExample`.)

## 2. Template mapping

```
normalize -> count(-) on G_t -> commit -> count(+) on G_{t+1} -> finalize
```

`detail::triangle_delta_problem` (`cpp/src/algorithms/triangle_delta/problem.hpp`, the template
card at its top) says what happens when; `triangle_delta_engine_impl<exec_t>` (`engine.hpp`) says
how, once for every backend (a `DYNG_HD` functor per element, run by the executors of
`cpp/src/operators/`).

| Step | Hook (profiler stage) | What it does | Operator (PLAN 4.5.3) |
|---|---|---|---|
| 0 | `normalize` (`triangle_delta.normalize`) | the net change of the batch as two lists of undirected edges (u, v), u < v, sorted, without repeats: the framework's normalized batch under `batch_semantics::set()` (ADR 0020), `graph/structural_change.hpp` on G_t otherwise; the position of an edge is its ownership id | `group_by_owner` (a sort) |
| 1a | `count` on the old view = count(-) (`triangle_delta.count_minus`) | for every deleted edge (u, v) with id i, the common neighbours w of u and v on G_t close triangles the batch destroys; the edge counts those whose other edges (u, w) and (v, w) are not deleted edges with an id below i | `count_delta<ownership::min_member>`, `intersect` (sorted merge) |
| apply | commit (`triangle_delta.commit`) | the batch applied once; no transposition (the counts read the out-edges, which an undirected graph stores both ways) | – |
| 2 | `count` on the new view = count(+) (`triangle_delta.count_plus`) | the same for every inserted edge on G_{t+1} | as count(-) |
| finish | `finalize` (`triangle_delta.finalize`) | count = count - removed + added (a removal larger than the count throws `internal_error`); the stats | – |

`compute()` is the static enactor: `triangle_delta.reset` -> `triangle_delta.count` (every
vertex u counts the triangles u < v < w through its row) -> `triangle_delta.finalize`.

Invariants: the subtraction runs on an `old_view` before the commit and the addition on a
`new_view` after it (I1); the problem names its ownership rule `ownership::min_member`, and the
framework passes it to both counts (I2); the counts are 64-bit (I7).

The four Chapter 3 challenges: (i) **affected set**: the changed edges (a triangle changes only
if it contains one); (ii) **propagation scope**: one intersection of two rows per changed edge, no
iteration; (iii) **correctness under parallelism**: ownership makes the per-edge counts independent
(each triangle is counted by its smallest changed edge only), and each count adds to one 64-bit
counter with an atomic addition; (iv) **data structure**: sorted, symmetric rows (the
intersection merges them).

## 3. API

```cpp
#include <dyng/triangle_delta.hpp>

dyng::graph_properties props;
props.directed = false;  // sorted rows and no parallel edges are the defaults
auto g = dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>::from_edges(res, list.view(), props);
auto tri = dyng::triangle_delta::compute(res, g);
dyng::triangle_delta::stats s = dyng::triangle_delta::update(res, g, batch.view(), tri);
// tri.count(), s.triangles_removed, s.triangles_added
```

```python
import dyng
g = dyng.Graph.from_edges([0, 0, 0, 1, 1], [1, 2, 3, 2, 3], directed=False)
tri = dyng.triangle_delta.compute(g)
stats = dyng.triangle_delta.update(g, dyng.EdgeBatch(insert=([2], [3])), tri)
print(tri.count, stats.triangles_added)   # 4 2
```

No options yet. Stats, all deterministic:

| Stat | Meaning |
|---|---|
| `affected` | 1 if the count changed, else 0 |
| `deletions`, `insertions` | the undirected edges of the two change lists |
| `triangles_removed`, `triangles_added` | what count(-) and count(+) found |
| `frontier_visits` | `deletions + insertions` |
| `batch` | what applying the batch did to the graph (each stored direction counts) |

Graph types: `graph<int32_t, int32_t or int64_t, int32_t or unweighted>` and
`graph<int64_t, int64_t, int32_t>`. `dyng::update(res, g, batch, r1, r2, ...)` accepts
`triangle_delta` results in C++; the Python `dyng.update()` and the CLI do not (ADR 0033).

## 4. Backends, engines and determinism

One engine (Tier A, `engine::operators`) on all three backends; the count and every counter are
identical on every backend and run (C3, C6). On cuda the counts read the graph's resident device
copy (G_t before the commit, G_{t+1} after it), and each count reads its counter back: at most two
host synchronizations per update (C8).

## 5. Complexity and performance notes

An update costs one sorted-row intersection per changed edge, plus a binary search of the change
list per common neighbour (the ownership test); compute() one intersection per edge. The profiler
stages are the hooks of the table above. There is no benchmark suite and no performance gate:
`triangle_delta` is teaching material.

## 6. Limitations

- Undirected simple graphs only (a directed graph is rejected with `invalid_argument_error`).
- Not tuned: one thread per changed edge (or vertex), no load balancing of high-degree vertices, no
  bitmap intersection.

## 7. Differences from the paper

Not from a paper: a teaching algorithm written for dynG (PLAN Section 9.3).

## 8. Mapping from the original code

None (not a port).

## 9. How to cite

`dyng::citation("triangle_delta")` returns the library's BibTeX entry (the manifest cites no
paper).
