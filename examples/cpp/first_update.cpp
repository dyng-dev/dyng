// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// first_update: the program of docs/getting_started/first_update_cpp.md. It builds a four-vertex
// graph, computes shortest paths from vertex 0, applies one batch with sssp::update() and prints
// the statistics and the repaired tree. CTest runs it on every CPU backend and checks the output.
//
//   first_update [sequential|openmp]
#include <dyng/dyng.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  const std::string backend = argc > 1 ? argv[1] : "openmp";
  auto res = backend == "sequential" ? dyng::resources::sequential() : dyng::resources::openmp();

  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 1;
  edges.add_edge(0, 1, {4});
  edges.add_edge(0, 2, {1});
  edges.add_edge(2, 1, {2});
  edges.add_edge(1, 3, {1});
  auto g = dyng::graph<std::int32_t, std::int64_t, std::int32_t>::from_edges(res, edges.view());

  auto tree = dyng::sssp::compute(res, g, /*source=*/0);  // canonical tree: lowest-id ties

  dyng::edge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_edge(2, 1);
  batch.insert_edge(2, 3, {1});
  dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);  // applies the batch

  std::printf("invalidated %lld, affected %lld\n", static_cast<long long>(st.invalidated),
              static_cast<long long>(st.affected));
  std::printf("distances");
  for (auto d : tree.distances()) std::printf(" %lld", static_cast<long long>(d));
  std::printf("\nparents");
  for (auto p : tree.parents()) std::printf(" %lld", static_cast<long long>(p));
  std::printf("\n");  // distances 0 4 1 2, parents -1 0 0 2
}
