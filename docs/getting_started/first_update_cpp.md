# A first update in C++

This program builds a four-vertex graph, computes shortest paths from vertex 0, and then applies
a batch with `sssp::update()`, which changes the graph **and** brings the result up to date.

```cpp
#include <dyng/dyng.hpp>

#include <cstdint>
#include <cstdio>

int main() {
  auto res = dyng::resources::openmp(8);   // or dyng::resources::sequential()

  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 1;
  edges.add_edge(0, 1, {4});
  edges.add_edge(0, 2, {1});
  edges.add_edge(2, 1, {2});
  edges.add_edge(1, 3, {1});
  auto g = dyng::graph<std::int32_t, std::int64_t, std::int32_t>::from_edges(res, edges.view());

  auto tree = dyng::sssp::compute(res, g, /*source=*/0);   // canonical tree: lowest-id ties

  dyng::edge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_edge(2, 1);
  batch.insert_edge(2, 3, {1});
  dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);

  // tree.distances() == {0, 4, 1, 2}, tree.parents() == {-1, 0, 0, 2}
  std::printf("invalidated %lld, affected %lld\n", static_cast<long long>(st.invalidated),
              static_cast<long long>(st.affected));
}
```

What happened:

1. `resources` chose the backend (Explanation: {doc}`../concepts/backends_and_resources`).
2. `compute()` produced the static result, the canonical shortest-path tree.
3. `update()` applied the batch to `g` (its version went up by one) and repaired the tree:
   the deletion of the tree edge (2, 1) invalidated the subtree below vertex 1, and the
   insertion (2, 3) gave vertex 3 a shorter path. The result now matches `g` again, so a
   later `update()` accepts it ({doc}`../concepts/results_and_versions`).

Build it against an installed dynG with `find_package(dyng)` ({doc}`install`). The next step is
the tutorial {doc}`../tutorials/sssp_mosp_files`, which runs the same steps on the file formats
of the original research code.
