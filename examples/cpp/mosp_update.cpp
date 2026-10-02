// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// mosp_update: build a graph with K objectives from MOSP text files, compute the K trees, the
// MOSP tree and the path costs with a preference vector, apply a batch with mosp::update() and
// write the combined graph's outputs (byte-compatible with MOSP's combinedGraph/ files).
//
//   mosp_update <csrPrefix> <insert.txt> <delete.txt> <outDir> <p1,..,pK> [sequential|openmp|cuda]
//
// With `cuda` the trees and the combined arrays live on the GPU (device 0); the program exits
// with 77 when the library has no CUDA backend or no device is visible.
#include <dyng/dyng.hpp>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;  // the originals' types

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: mosp_update <csrPrefix> <insert.txt> <delete.txt> <outDir> <p1,..,pK> "
                 "[backend]\n";
    return 2;
  }
  const std::string backend = argc > 6 ? argv[6] : "sequential";
  if (backend != "sequential" && backend != "openmp" && backend != "cuda") {
    std::cerr << "mosp_update: unknown backend '" << backend << "' (sequential, openmp, cuda)\n";
    return 2;
  }
  if (backend == "cuda" && !dyng::backend_available(dyng::backend::cuda)) {
    std::cerr << "mosp_update: no CUDA backend or no visible device\n";
    return 77;
  }
  dyng::resources res = backend == "openmp" ? dyng::resources::openmp()
                        : backend == "cuda" ? dyng::resources::cuda()
                                            : dyng::resources::sequential();

  const auto csr = dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(argv[1]);
  graph_t g = graph_t::from_csr(res, csr.view(), dyng::graph_properties::mosp_compatible());

  dyng::mosp::options opt;
  std::istringstream prefs(argv[5]);
  for (std::string p; std::getline(prefs, p, ',');) {
    opt.preferences.push_back(static_cast<std::int32_t>(std::stol(p)));
  }
  auto paths = dyng::mosp::compute(res, g, /*source=*/0, opt);  // K trees, MOSP tree, costs

  dyng::io::legacy_batch_options options;
  options.num_weights = g.num_weights();
  options.num_vertices = g.num_vertices();
  const auto batch =
      dyng::io::read_legacy_batch<std::int32_t, std::int32_t>(argv[2], argv[3], options);
  const dyng::mosp::stats st = dyng::mosp::update(res, g, batch.view(), paths);
  std::printf("K=%d L=%lld: combined edges %lld, affected %lld\n", paths.num_objectives(),
              static_cast<long long>(st.preference_scale),
              static_cast<long long>(st.combined_edges), static_cast<long long>(st.affected));

  // The writers take host arrays; to_vector() copies (from device memory on cuda). The path
  // costs are host memory on every backend.
  const std::vector<std::int64_t> distances = dyng::to_vector(res, paths.combined_distances());
  const std::vector<std::int32_t> parents = dyng::to_vector(res, paths.combined_parents());
  const std::string out = argv[4];
  dyng::io::write_distances(out + "/distancesCsr.txt", dyng::host_view(distances));
  dyng::io::write_parents(out + "/SSSPTreeCsr.txt", dyng::host_view(parents));
  dyng::io::write_path_costs(out + "/mospCosts.txt", paths.path_costs(), paths.num_objectives());
  return 0;
}
