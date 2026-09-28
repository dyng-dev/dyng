// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// cycle_count_update: count the directed cycles of length 2..k of an edge-list graph (a TUDataset
// *_A.txt file loads directly), apply a generated batch with cycle_count::update() and print the
// updated histogram in CycleEnumeration-GPU's CSV format.
//
//   cycle_count_update <edges.txt> <k> <deletions> <insertions> <seed> [sequential|openmp]
#include <dyng/dyng.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

using graph_t = dyng::graph<std::int32_t, std::int64_t, dyng::unweighted>;

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: cycle_count_update <edges.txt> <k> <deletions> <insertions> <seed> "
                 "[backend]\n";
    return 2;
  }
  const std::string backend = argc > 6 ? argv[6] : "sequential";
  dyng::resources res =
      backend == "openmp" ? dyng::resources::openmp() : dyng::resources::sequential();

  const auto edges = dyng::io::read_edge_list<std::int32_t, dyng::unweighted>(argv[1]);
  const auto props = dyng::graph_properties::cycle_enum_compatible();  // sorted rows, set()
  graph_t g = graph_t::from_edges(res, edges.view(), props);

  dyng::cycle_count::options opt;
  opt.max_length = std::atoi(argv[2]);                  // cycles of length 2..k
  auto hist = dyng::cycle_count::compute(res, g, opt);  // exact static count (Johnson)

  dyng::generators::legacy::cycle_enum_batch_options bo;  // CycleEnum's generate_batch
  bo.num_deletions = std::atoll(argv[3]);
  bo.num_insertions = std::atoll(argv[4]);
  bo.seed = std::strtoull(argv[5], nullptr, 10);
  const auto batch = dyng::generators::legacy::cycle_enum_batch(g.to_csr(res).view(), bo);

  // Subtract the cycles through the deletions on G_t, apply, add those through the insertions.
  const dyng::cycle_count::stats st = dyng::cycle_count::update(res, g, batch.view(), hist);
  std::cerr << "-" << st.deletions << " +" << st.insertions << " edges: " << st.cycles_removed
            << " cycles removed, " << st.cycles_added << " added\n";
  for (int len = 2; len <= opt.max_length; ++len) {
    std::cerr << len << "-cycles: " << hist.count(len) << '\n';
  }
  dyng::io::write_histogram_csv(std::cout, hist.counts());  // "# cycle_size, ..." "Total, N"
  return 0;
}
