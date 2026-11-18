// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// label_propagation_stream: DynLP on LabelPropagation-CUDA's vertex-arrival stream, the published
// setup (a graph with the stream's row capacities, compute() on the empty graph, one update() per
// batch), and the labels written as the original's dynlp_labels.txt.
#include <dyng/dyng.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
namespace lp = dyng::label_propagation;
using graph_t = dyng::graph<std::int32_t, std::int32_t, float>;

int main(int argc, char** argv) try {  // exit 1: a wrong argument or a failed call; 77: no GPU
  const std::string b = argc > 2 ? argv[2] : "sequential";  // the backend (no OpenMP one)
  if (argc < 2 || argc > 4 || (b != "sequential" && b != "cuda"))
    throw dyng::invalid_argument_error("usage: <out_dir> [sequential|cuda] [total_vertices]");
  if (b == "cuda" && !dyng::backend_available(dyng::backend::cuda)) return 77;  // skipped
  const dyng::resources res = b == "cuda" ? dyng::resources::cuda() : dyng::resources::sequential();
  dyng::generators::vertex_arrival_options o;  // the original's --numBatches 5 --seed 42
  o.total_vertices = argc > 3 ? std::stoll(argv[3]) : 5000;
  o.num_batches = 5;
  const auto stream =
      dyng::generators::legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), o);
  dyng::graph_properties p;  // slot rows with the full degrees: the original's device graph
  p.directed = false;
  p.layout = dyng::row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  graph_t g = graph_t::with_capacity(res, static_cast<std::int32_t>(o.total_vertices),
                                     dyng::host_view(stream.row_capacity), p);
  lp::options opt;
  opt.update_schedule = lp::schedule::damped;                     // bitwise per backend
  opt.similarity_threshold = stream.similarity_threshold;         // the published tau
  lp::result<std::int32_t> r = lp::compute(res, g, {}, {}, opt);  // every label 0.5
  r.reserve(res, stream.max_batch_vertices, stream.max_batch_edges);
  for (std::size_t i = 0; i < stream.batches.size(); ++i) {
    const lp::stats st = lp::update(res, g, stream.batches[i].view(), r);
    std::cout << "batch " << i << ": " << st.iterations << " iterations, " << st.components
              << " components, " << g.num_active_vertices() << " active\n";
  }
  const std::vector<float> labels = dyng::to_vector(res, r.labels());  // host copy (cuda)
  dyng::io::write_labels(std::string(argv[1]) + "/dynlp_labels.txt", dyng::host_view(labels));
} catch (const dyng::error& e) {
  std::cerr << "label_propagation_stream: " << e.what() << '\n';
  return 1;
}
