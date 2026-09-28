// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// first_update: the program of docs/getting_started/first_update_cpp.md. It builds a four-vertex
// graph, computes shortest paths from vertex 0, applies one batch with sssp::update() and prints
// the statistics and the repaired tree. CTest runs it on every backend and checks the output.
//
//   first_update [sequential|openmp|cuda]
//
// With `cuda` the graph and the tree live on GPU 0; the program exits with 77 (a skipped test)
// when the library has no CUDA backend or no device is visible.
#include <dyng/dyng.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string backend = argc > 1 ? argv[1] : "openmp";
  if (backend == "cuda" && !dyng::backend_available(dyng::backend::cuda)) {
    std::printf("no CUDA device visible\n");
    return 77;
  }
  auto res = backend == "sequential" ? dyng::resources::sequential()
             : backend == "cuda"     ? dyng::resources::cuda()
                                     : dyng::resources::openmp();

  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 1;
  edges.add_edge(0, 1, {4});
  edges.add_edge(0, 2, {1});
  edges.add_edge(2, 1, {2});
  edges.add_edge(1, 3, {1});
  auto g = dyng::graph<>::from_edges(res, edges.view());  // int32 ids, offsets and weights

  auto tree = dyng::sssp::compute(res, g, /*source=*/0);  // canonical tree: lowest-id ties

  dyng::edge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_edge(2, 1);
  batch.insert_edge(2, 3, {1});
  dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);  // applies the batch

  std::printf("invalidated %lld, affected %lld\n", static_cast<long long>(st.invalidated),
              static_cast<long long>(st.affected));
  // to_vector() copies a result to the host (from device memory on cuda).
  const std::vector<std::int64_t> distances = dyng::to_vector(res, tree.distances());
  const std::vector<std::int32_t> parents = dyng::to_vector(res, tree.parents());
  std::printf("distances");
  for (auto d : distances) std::printf(" %lld", static_cast<long long>(d));
  std::printf("\nparents");
  for (auto p : parents) std::printf(" %lld", static_cast<long long>(p));
  std::printf("\n");  // distances 0 4 1 2, parents -1 0 0 2
}
