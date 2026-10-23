// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// triad_count_update: count the triads of a .hg hypergraph by h-motif (30 bins) and StatHyper's
// type 1, then apply the batches of a .dgt file to both results with one dyng::update() each.
#include <dyng/dyng.hpp>

#include <cstdint>
#include <iostream>
#include <string>
using hypergraph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
namespace tc = dyng::triad_count;
template <typename values_t>  // the non-zero bins as bin:value (bin k is ESCHER's h-motif k + 1)
void print_bins(const char* label, const values_t& values, int num_bins) {
  std::cout << label;
  for (int k = 0; k < num_bins; ++k)
    if (values[k] != 0) std::cout << ' ' << k << ':' << values[k];
  std::cout << '\n';
}

int main(int argc, char** argv) try {  // exit 1: a wrong argument or a failed call; 77: no GPU
  const std::string b = argc > 3 ? argv[3] : "sequential";  // the backend
  if (argc < 3 || argc > 4 || (b != "sequential" && b != "openmp" && b != "cuda"))
    throw dyng::invalid_argument_error("usage: <file.hg> <file.dgt> [sequential|openmp|cuda]");
  if (b == "cuda" && !dyng::backend_available(dyng::backend::cuda)) return 77;  // skipped
  dyng::resources res = b == "openmp" ? dyng::resources::openmp()
                        : b == "cuda" ? dyng::resources::cuda()
                                      : dyng::resources::sequential();
  // [tutorial: read]
  dyng::io::hypergraph_read_options ro;
  ro.dedupe_hyperedges = true;  // repeated hyperedges dropped; file ids 1..n become 0..n-1
  const auto hes = dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>(argv[1], ro);
  dyng::hypergraph_properties hp;
  hp.line_graph = dyng::line_graph_policy::sorted_merge;  // h_motif reads the line graph
  hypergraph_t hg = hypergraph_t::from_hyperedges(res, hes.view(), hp);
  // [tutorial: read end]
  // [tutorial: compute]
  tc::result motifs = tc::compute(res, hg);  // kind = pattern::h_motif: 30 bins
  tc::options o;
  o.kind = tc::pattern::stathyper_type1;  // one bin: the sum of C(|e|, 3)
  tc::result type1 = tc::compute(res, hg, o);
  // [tutorial: compute end]
  std::cout << hg.num_hyperedges() << " hyperedges: " << motifs.total() << " triads, type1 "
            << type1.total() << '\n';
  print_bins("  bins", motifs.counts(), motifs.num_bins());
  const auto batches = dyng::io::read_hyperedge_batches<std::int32_t, dyng::unweighted>(argv[2]);
  for (std::size_t i = 0; i < batches.size(); ++i) {
    // [tutorial: update]
    auto [st, st1] = dyng::update(res, hg, batches[i].view(), motifs, type1);  // one commit
    std::cout << "batch " << i << ": " << st.deleted_frontier << " deleted, inserted ids";
    for (const std::int64_t id : st.batch.inserted_ids) std::cout << ' ' << id;
    std::cout << "\n  h_motif -" << st.triads_removed << " +" << st.triads_added << " = "
              << motifs.total() << ", type1 -" << st1.triads_removed << " +" << st1.triads_added
              << " = " << type1.total() << '\n';
    print_bins("  delta", st.delta, motifs.num_bins());
    // [tutorial: update end]
  }
} catch (const dyng::error& e) {  // the message names a reader's file, line and column
  std::cerr << "triad_count_update: " << e.what() << '\n';
  return 1;
}
