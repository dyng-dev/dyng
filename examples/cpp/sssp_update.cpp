// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// sssp_update: build a graph from MOSP text files, compute a shortest-path tree, apply a batch
// with sssp::update() and write the updated tree (byte-compatible with MOSP's output files).
//
//   sssp_update <csrPrefix> <insert.txt> <delete.txt> <outDir> [sequential|openmp|cuda]
//
// With `cuda` the graph and the tree live on the GPU (device 0) and the fused engine runs the
// update; the program exits with 77 when the library has no CUDA backend or no device is visible.
#include <dyng/dyng.hpp>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;  // the originals' types

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: sssp_update <csrPrefix> <insert.txt> <delete.txt> <outDir> [backend]\n";
    return 2;
  }
  const std::string backend = argc > 5 ? argv[5] : "sequential";
  if (backend != "sequential" && backend != "openmp" && backend != "cuda") {
    std::cerr << "sssp_update: unknown backend '" << backend << "' (sequential, openmp, cuda)\n";
    return 2;
  }
  if (backend == "cuda" && !dyng::backend_available(dyng::backend::cuda)) {
    std::cerr << "sssp_update: no CUDA backend or no visible device\n";
    return 77;
  }
  dyng::resources res = backend == "openmp" ? dyng::resources::openmp()
                        : backend == "cuda" ? dyng::resources::cuda()
                                            : dyng::resources::sequential();

  const auto csr = dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(argv[1]);
  graph_t g = graph_t::from_csr(res, csr.view(), dyng::graph_properties::mosp_compatible());

  auto tree = dyng::sssp::compute(res, g, /*source=*/0);  // canonical tree (lowest-id ties)

  dyng::io::legacy_batch_options options;
  options.num_weights = g.num_weights();
  options.num_vertices = g.num_vertices();
  const auto batch =
      dyng::io::read_legacy_batch<std::int32_t, std::int32_t>(argv[2], argv[3], options);

  dyng::profiler prof;
  res.attach_profiler(&prof);
  const dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);
  res.attach_profiler(nullptr);
  std::printf("+%lld -%lld edges, invalidated %lld, affected %lld, engine %s\n",
              static_cast<long long>(st.batch.inserted_edges),
              static_cast<long long>(st.batch.deleted_edges),
              static_cast<long long>(st.invalidated), static_cast<long long>(st.affected),
              st.engine_used == dyng::engine::fused ? "fused" : "operators");

  // The writers take host arrays; to_vector() copies (from device memory on cuda).
  const std::vector<std::int64_t> distances = dyng::to_vector(res, tree.distances());
  const std::vector<std::int32_t> parents = dyng::to_vector(res, tree.parents());
  const dyng::array_view<const std::int64_t> host_distances(distances.data(), distances.size());
  const dyng::array_view<const std::int32_t> host_parents(parents.data(), parents.size());
  const std::string out = argv[4];
  dyng::io::write_distances(out + "/distancesUpdated.txt", host_distances);
  dyng::io::write_parents(out + "/SSSPTreeUpdated.txt", host_parents);
  prof.write_csv(std::cout);  // sssp.update, sssp.prepare, sssp.commit, sssp.loop, ...
  return 0;
}
