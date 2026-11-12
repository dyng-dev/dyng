// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// hyper_sssp_update: shortest hyperpaths of a weighted .hg hypergraph from two source vertices,
// kept up to date by one dyng::update() per batch of a .dgt file (hyperedges and incidence edits).
#include <dyng/dyng.hpp>

#include <cstdint>
#include <iostream>
#include <string>
using hypergraph_t = dyng::hypergraph<std::int32_t, std::int32_t>;
namespace hs = dyng::hyper_sssp;
void print(const dyng::resources& res, const hs::result<std::int32_t>& r) {  // reached ids: id:D/p
  const auto d = dyng::to_vector(res, r.hyperedge_distances());
  const auto p = dyng::to_vector(res, r.hyperedge_parents());
  std::cout << "  from " << r.source() << ':';
  for (std::size_t h = 0; h < d.size(); ++h)
    if (d[h] < dyng::infinite_distance<std::int64_t>())
      std::cout << ' ' << h << ':' << d[h] << '/' << p[h];
  std::cout << " | vertex 11: " << r.distance_to(11) << '\n';
}

int main(int argc, char** argv) try {  // exit 1: a wrong argument or a failed call; 77: no GPU
  const std::string b = argc > 3 ? argv[3] : "sequential";  // no openmp backend in 0.4
  if (argc < 3 || argc > 4 || (b != "sequential" && b != "cuda"))
    throw dyng::invalid_argument_error("usage: <file.hg> <file.dgt> [sequential|cuda]");
  if (b == "cuda" && !dyng::backend_available(dyng::backend::cuda)) return 77;  // skipped
  const dyng::resources res = b == "cuda" ? dyng::resources::cuda() : dyng::resources::sequential();
  // [tutorial: read]
  const auto hes = dyng::io::read_hypergraph<std::int32_t, std::int32_t>(argv[1]);  // '; w'
  dyng::hypergraph_properties hp;
  hp.storage = dyng::hypergraph_storage::slack_csr;  // incidence edits and hyper_sssp need it
  hp.line_graph = dyng::line_graph_policy::slack_csr;
  hypergraph_t hg = hypergraph_t::from_hyperedges(res, hes.view(), hp);
  // [tutorial: read end]
  // [tutorial: compute]
  hs::result<std::int32_t> a = hs::compute(res, hg, 0);  // from vertex 0
  hs::options o;
  o.work_budget = 0.5;  // fall back to a recompute sooner (stats::relaxations shows the work)
  hs::result<std::int32_t> c = hs::compute(res, hg, 9, o);  // from vertex 9
  // [tutorial: compute end]
  std::cout << hg.num_hyperedges() << " hyperedges, " << hg.num_vertices() << " vertices\n";
  print(res, a);
  print(res, c);
  const auto batches = dyng::io::read_hyperedge_batches<std::int32_t, std::int32_t>(argv[2]);
  for (std::size_t i = 0; i < batches.size(); ++i) {
    // [tutorial: update]
    auto [sa, sc] = dyng::update(res, hg, batches[i].view(), a, c);  // one commit, both results
    std::cout << "batch " << i << ": -" << sa.batch.deleted_hyperedges << " +"
              << sa.batch.inserted_hyperedges << " hyperedges, incidences -"
              << sa.batch.deleted_incidences << " +" << sa.batch.inserted_incidences
              << ", line graph -" << sa.line_graph_deletions << " +" << sa.line_graph_insertions
              << ", invalidated " << sa.invalidated << " and " << sc.invalidated << '\n';
    // [tutorial: update end]
    print(res, a);
    print(res, c);
  }
} catch (const dyng::error& e) {  // the message names a reader's file, line and column
  std::cerr << "hyper_sssp_update: " << e.what() << '\n';
  return 1;
}
