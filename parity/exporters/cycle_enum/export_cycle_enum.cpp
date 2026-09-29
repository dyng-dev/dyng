// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// Graph/io/generator exporter of the parity harness for CycleEnumeration-GPU@0a976ad.
// parity/export_patches/CycleEnumeration-GPU/build.sh compiles it into the PATCHED scratch copy,
// against that copy's unchanged sources (src/core, src/dynamic, src/sequential and the header-only
// tests/support/cycle_oracles.hpp). It never touches the original repository and calls the original
// functions unchanged:
//
//   parse     <file> [--digest]      read_temporal_graph: external ids, grouped edges, timestamps
//   csr       <file> [--digest]      build_directed_graph(read_graph_view(file))
//   generate  <file> <del> <ins> <seed> [<window>] [--digest]
//                                    generate_batch(read_graph_view(file), params)
//   apply-generated <file> <del> <ins> <seed> [<window>] [--digest]
//                                    generate_batch, prepare_batch, apply_batch: the prepared batch
//                                    and the CSR of the graph after the batch
//   random-cases <dir> <count> <seed>
//                                    writes <count> random small cases (a graph and an arbitrary
//                                    batch each) as <dir>/case_<i>.txt
//   apply     <case file>            for one case: prepare_batch, apply_batch (checked equal for
//                                    the raw and the prepared batch), and the cycle histograms of
//                                    the graph before and after from the original's subset-DP
//                                    oracle and its brute force
//   counts    <case file> <threads>  for one case and every k in 2..7 and "no bound": the static
//                                    histograms of the graph before and after the batch from the
//                                    sequential Johnson and the OpenMP counter, and the updated
//                                    histogram of update_static_histogram and
//                                    update_static_histogram_openmp (bounded k only)
//   count-file <file> <k> <threads> [--omp-only] [--digest]
//                                    the static histograms of read_graph_view(file) (sequential
//                                    Johnson, unless --omp-only, and OpenMP counter; k = -1: no
//                                    bound)
//   update-file <file> <k> <del> <ins> <seed> <threads> [<window>] [--digest]
//                                    the prior (OpenMP counter), generate_batch, then
//                                    update_static_histogram and update_static_histogram_openmp
//
// Built a second time as export_cycle_enum_cuda (CYCLE_ENUM_CUDA_ENABLED=1, linked against the
// copy's CUDA libraries; M2b), it also has the original's CUDA backend:
//   cuda-counts <case file> <device>  for one case and every k in 2..7 and "no bound" (k = the
//                                    vertex count: the device counters need a bound): the static
//                                    histograms before and after the batch from the work-queue
//                                    counter with automatic items (count_simple_cycles_johnson_
//                                    work_queue), before also with every other scheduler (naive
//                                    count_simple_cycles_johnson; work-queue roots, edges, two-hop),
//                                    and update_static_histogram_cuda (bounded k only)
//   cuda-count-file <file> <k> <device> [--digest]
//                                    the same static histograms of read_graph_view(file)
//   cuda-update-file <file> <k> <del> <ins> <seed> <device> [<window>] [--digest]
//                                    the prior (work queue), generate_batch, then
//                                    update_static_histogram_cuda
//
// Text formats (the dynG tests produce the same text and compare it, or its digest):
//   parse:   "vertices <n>\nedges <m>\nevents <t>\n", "v <external id>\n" per compact vertex, then
//            "e <u> <v> <t1> <t2> ...\n" per grouped edge (timestamps sorted)
//   csr:     "n <n>\nm <m>\nrow_ptr <o0> <o1> ...\ncol_ind <c0> <c1> ...\n"
//   batch:   "- <u> <v>\n" per deletion, then "+ <u> <v>\n" per insertion
//   case:    "n <n>\n", "e <u> <v>\n" per edge, "- <u> <v>\n" per deletion and "+ <u> <v>\n" per
//            insertion, in batch order
//   apply:   "prepared\n" + batch, "after\n" + csr, then "oracle_before <len>:<count> ...\n",
//            "oracle_after ...\n", "brute_before ...\n", "brute_after ...\n" (unbounded length)
//   counts:  per k ("k <k>\n", k = -1 for no bound): "seq_before ...\n", "omp_before ...\n",
//            "seq_after ...\n", "omp_after ...\n" and for k >= 2 "seq_update ...\n",
//            "omp_update ...\n" (histogram text: "<name> <len>:<count> ..." over the non-zero
//            lengths)
//   count-file: "seq ...\n", "omp ...\n"; update-file: "prior ...\n", "seq_update ...\n",
//            "omp_update ...\n", with "deletions <d> insertions <i>\n" first
//   cuda-counts: per k ("k <k>\n"): "cuda_before ...\n", "cuda_naive_before ...\n",
//            "cuda_roots_before ...\n", "cuda_edges_before ...\n", "cuda_two_hop_before ...\n",
//            "cuda_after ...\n" and for k >= 2 "cuda_update ...\n"
//   cuda-count-file: "cuda ...\n", "cuda_naive ...\n", "cuda_roots ...\n", "cuda_edges ...\n",
//            "cuda_two_hop ...\n"; cuda-update-file: "deletions <d> insertions <i>\n",
//            "prior ...\n", "cuda_update ...\n"
// --digest prints "fnv1a64 <16 hex digits> bytes <size>" of the text instead of the text.
//
// The program is not part of the dynG build; it exists so that the committed fixtures and digests
// can be regenerated from the pinned original at any time.

#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/dynamic/batch_generator.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"
#include "cycle_enum/dynamic/update_openmp.hpp"
#include "cycle_enum/dynamic/update_sequential.hpp"
#include "cycle_enum/openmp/openmp_johnson.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/johnson.hpp"
#include "support/cycle_oracles.hpp"

#ifndef CYCLE_ENUM_CUDA_ENABLED
#define CYCLE_ENUM_CUDA_ENABLED 0
#endif
#if CYCLE_ENUM_CUDA_ENABLED
#include "cycle_enum/cuda/cuda_johnson.hpp"
#include "cycle_enum/cuda/cuda_work_queue.hpp"
#include "cycle_enum/dynamic/update_cuda.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace dyn = cycle_enum::dynamic;

int usage() {
  std::cerr << "usage: export_cycle_enum parse|csr|generate|apply-generated|random-cases|apply|"
               "counts|count-file|update-file|cuda-counts|cuda-count-file|cuda-update-file ...\n";
  return 2;
}

std::uint64_t fnv1a64(const std::string& text) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const char c : text) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

void emit(const std::string& text, const bool digest) {
  if (digest) {
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a64(text)));
    std::cout << "fnv1a64 " << hex << " bytes " << text.size() << "\n";
  } else {
    std::cout << text;
  }
}

std::string csr_text(const dyn::DirectedGraph& g) {
  std::ostringstream out;
  out << "n " << g.vertex_count << "\nm " << g.neighbors.size() << "\nrow_ptr";
  for (const std::size_t o : g.offsets) {
    out << ' ' << o;
  }
  out << "\ncol_ind";
  for (const cycle_enum::VertexId v : g.neighbors) {
    out << ' ' << v;
  }
  out << '\n';
  return out.str();
}

std::string batch_text(const dyn::EdgeBatch& b) {
  std::ostringstream out;
  for (const dyn::EdgeChange& c : b.deletions) {
    out << "- " << c.source << ' ' << c.target << '\n';
  }
  for (const dyn::EdgeChange& c : b.insertions) {
    out << "+ " << c.source << ' ' << c.target << '\n';
  }
  return out.str();
}

std::string histogram_text(const char* name, const cycle_enum::CycleHistogram& h) {
  std::ostringstream out;
  out << name;
  for (const auto& [length, count] : h.entries()) {
    out << ' ' << length << ':' << count;
  }
  out << '\n';
  return out.str();
}

std::string parse_text(const cycle_enum::TemporalGraph& g) {
  std::ostringstream out;
  out << "vertices " << g.vertex_count() << "\nedges " << g.edge_count() << "\nevents "
      << g.timestamp_count() << '\n';
  for (std::size_t v = 0; v < g.vertex_count(); ++v) {
    out << "v " << g.external_id(static_cast<cycle_enum::VertexId>(v)) << '\n';
  }
  for (const cycle_enum::TemporalEdge& e : g.edges()) {
    out << "e " << e.source << ' ' << e.target;
    for (const cycle_enum::Timestamp t : e.timestamps) {
      out << ' ' << t;
    }
    out << '\n';
  }
  return out.str();
}

dyn::BatchParams batch_params(char** argv, const int argc, const int first, bool& digest) {
  dyn::BatchParams params;
  params.num_deletions = std::strtoull(argv[first], nullptr, 10);
  params.num_insertions = std::strtoull(argv[first + 1], nullptr, 10);
  params.seed = std::strtoull(argv[first + 2], nullptr, 10);
  for (int i = first + 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--digest") {
      digest = true;
    } else {
      params.locality_window = std::strtoull(argv[i], nullptr, 10);
    }
  }
  return params;
}

struct test_case {
  std::size_t n = 0;
  std::vector<cycle_enum::TemporalEdge> edges;
  dyn::EdgeBatch batch;
};

test_case read_case(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  test_case c;
  std::string tag;
  while (in >> tag) {
    if (tag == "n") {
      in >> c.n;
    } else {
      std::uint64_t u = 0;
      std::uint64_t v = 0;
      in >> u >> v;
      const auto su = static_cast<cycle_enum::VertexId>(u);
      const auto sv = static_cast<cycle_enum::VertexId>(v);
      if (tag == "e") {
        c.edges.push_back({su, sv, {0}});
      } else if (tag == "-") {
        c.batch.deletions.push_back({su, sv});
      } else if (tag == "+") {
        c.batch.insertions.push_back({su, sv});
      } else {
        throw std::runtime_error("bad tag " + tag + " in " + path);
      }
    }
  }
  return c;
}

/// Arbitrary batches as in the original's UpdateBatchValidationTest: endpoints up to two ids past
/// the graph, coinciding endpoints, deletions of absent and insertions of present edges, duplicates
/// and delete-then-reinsert pairs.
void random_cases(const std::string& dir, const int count, const std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  for (int i = 0; i < count; ++i) {
    namespace ts = cycle_enum::test_support;
    ts::RandomGraphSpec spec = ts::random_spec(rng, 2, 11, false);
    spec.self_loop_probability = 0.0;  // the parser never produces self-loops
    const std::vector<cycle_enum::TemporalEdge> edges = ts::random_edges(spec, rng);
    std::uniform_int_distribution<cycle_enum::VertexId> vertex(
        0, static_cast<cycle_enum::VertexId>(spec.vertex_count + 1));
    std::ostringstream out;
    out << "n " << spec.vertex_count << '\n';
    for (const cycle_enum::TemporalEdge& e : edges) {
      out << "e " << e.source << ' ' << e.target << '\n';
    }
    std::vector<std::pair<cycle_enum::VertexId, cycle_enum::VertexId>> deletions;
    const std::size_t num_del = rng() % 9;
    const std::size_t num_ins = rng() % 9;
    for (std::size_t j = 0; j < num_del; ++j) {
      if (!edges.empty() && rng() % 2 == 0) {  // an existing edge
        const auto& e = edges[rng() % edges.size()];
        deletions.emplace_back(e.source, e.target);
      } else {
        deletions.emplace_back(vertex(rng), vertex(rng));
      }
    }
    for (const auto& [u, v] : deletions) {
      out << "- " << u << ' ' << v << '\n';
    }
    for (std::size_t j = 0; j < num_ins; ++j) {
      cycle_enum::VertexId u = 0;
      cycle_enum::VertexId v = 0;
      if (!edges.empty() && rng() % 4 == 0) {  // an existing edge
        const auto& e = edges[rng() % edges.size()];
        u = e.source;
        v = e.target;
      } else {
        u = vertex(rng);
        v = vertex(rng);
      }
      out << "+ " << u << ' ' << v << '\n';
      if (rng() % 5 == 0) {
        out << "+ " << u << ' ' << v << '\n';  // a duplicate
      }
    }
    if (i % 3 == 0 && !deletions.empty()) {  // delete, then re-insert
      out << "+ " << deletions.front().first << ' ' << deletions.front().second << '\n';
    }
    char name[64];
    std::snprintf(name, sizeof(name), "/case_%03d.txt", i);
    std::ofstream file(dir + name);
    file << out.str();
  }
}

std::string apply_case(const std::string& path) {
  namespace ts = cycle_enum::test_support;
  const test_case c = read_case(path);
  const cycle_enum::GraphView view0 = ts::view_from_edges(c.edges, c.n);
  const dyn::DirectedGraph g0 = dyn::build_directed_graph(view0);
  const dyn::EdgeBatch prepared = dyn::prepare_batch(g0, c.batch);
  const dyn::DirectedGraph raw = dyn::apply_batch(g0, c.batch);
  const dyn::DirectedGraph via_prepared = dyn::apply_batch(g0, prepared);
  if (csr_text(raw) != csr_text(via_prepared)) {
    throw std::runtime_error("apply_batch(raw) != apply_batch(prepared) for " + path);
  }
  const cycle_enum::GraphView view1 = dyn::to_graph_view(raw);
  std::string out = "prepared\n" + batch_text(prepared) + "after\n" + csr_text(raw);
  out += histogram_text("oracle_before", ts::oracle_simple_cycles(view0));
  out += histogram_text("oracle_after", ts::oracle_simple_cycles(view1));
  out +=
      histogram_text("brute_before", cycle_enum::sequential::count_simple_cycles_bruteforce(view0));
  out +=
      histogram_text("brute_after", cycle_enum::sequential::count_simple_cycles_bruteforce(view1));
  return out;
}

std::optional<std::size_t> bound_of(const long long k) {
  return k < 0 ? std::nullopt : std::optional<std::size_t>(static_cast<std::size_t>(k));
}

std::string counts_case(const std::string& path, const int threads) {
  namespace ts = cycle_enum::test_support;
  const test_case c = read_case(path);
  const cycle_enum::GraphView view0 = ts::view_from_edges(c.edges, c.n);
  const dyn::DirectedGraph g0 = dyn::build_directed_graph(view0);
  const cycle_enum::GraphView view1 = dyn::to_graph_view(dyn::apply_batch(g0, c.batch));
  std::string out;
  for (const long long k : {2LL, 3LL, 4LL, 5LL, 6LL, 7LL, -1LL}) {
    const std::optional<std::size_t> bound = bound_of(k);
    out += "k " + std::to_string(k) + "\n";
    const cycle_enum::CycleHistogram before =
        cycle_enum::sequential::count_simple_cycles_johnson(view0, bound);
    out += histogram_text("seq_before", before);
    out += histogram_text("omp_before",
                          cycle_enum::openmp::count_simple_cycles_johnson(view0, threads, bound));
    out += histogram_text("seq_after",
                          cycle_enum::sequential::count_simple_cycles_johnson(view1, bound));
    out += histogram_text("omp_after",
                          cycle_enum::openmp::count_simple_cycles_johnson(view1, threads, bound));
    if (bound) {
      out +=
          histogram_text("seq_update", dyn::update_static_histogram(g0, before, c.batch, *bound));
      out += histogram_text(
          "omp_update", dyn::update_static_histogram_openmp(g0, before, c.batch, *bound, threads));
    }
  }
  return out;
}

std::string count_file(const std::string& path, const long long k, const int threads,
                       const bool omp_only) {
  const cycle_enum::GraphView view = cycle_enum::read_graph_view(path);
  std::string out;
  if (!omp_only) {
    out += histogram_text("seq",
                          cycle_enum::sequential::count_simple_cycles_johnson(view, bound_of(k)));
  }
  return out + histogram_text("omp", cycle_enum::openmp::count_simple_cycles_johnson(view, threads,
                                                                                     bound_of(k)));
}

std::string update_file(const std::string& path, const std::size_t k,
                        const dyn::BatchParams& params, const int threads) {
  const cycle_enum::GraphView view = cycle_enum::read_graph_view(path);
  // The prior from the OpenMP counter (count-file checks it equal to the sequential Johnson).
  const cycle_enum::CycleHistogram prior =
      cycle_enum::openmp::count_simple_cycles_johnson(view, threads, k);
  const dyn::DirectedGraph g0 = dyn::build_directed_graph(view);
  const dyn::EdgeBatch batch = dyn::generate_batch(view, params);
  return "deletions " + std::to_string(batch.deletions.size()) + " insertions " +
         std::to_string(batch.insertions.size()) + "\n" + histogram_text("prior", prior) +
         histogram_text("seq_update", dyn::update_static_histogram(g0, prior, batch, k)) +
         histogram_text("omp_update",
                        dyn::update_static_histogram_openmp(g0, prior, batch, k, threads));
}

#if CYCLE_ENUM_CUDA_ENABLED
namespace cu = cycle_enum::cuda;

/// The static histograms of every CUDA scheduler, "<prefix><name><suffix> ..." lines; the first
/// (work queue, automatic items) is returned for the update.
std::string cuda_static(const cycle_enum::GraphView& view, const std::size_t k, const int device,
                        const std::string& prefix, const std::string& suffix,
                        cycle_enum::CycleHistogram* first) {
  const cycle_enum::CycleHistogram automatic =
      cu::count_simple_cycles_johnson_work_queue(view, device, k, nullptr, cu::CudaWorkItems::Auto);
  if (first != nullptr) {
    *first = automatic;
  }
  std::string out = histogram_text((prefix + suffix).c_str(), automatic);
  out += histogram_text((prefix + "_naive" + suffix).c_str(),
                        cu::count_simple_cycles_johnson(view, device, k, nullptr));
  const std::pair<const char*, cu::CudaWorkItems> kinds[] = {
      {"_roots", cu::CudaWorkItems::Roots},
      {"_edges", cu::CudaWorkItems::Edges},
      {"_two_hop", cu::CudaWorkItems::TwoHop}};
  for (const auto& [name, items] : kinds) {
    out +=
        histogram_text((prefix + name + suffix).c_str(),
                       cu::count_simple_cycles_johnson_work_queue(view, device, k, nullptr, items));
  }
  return out;
}

/// The effective bound of "no bound" on the device: the vertex count (at least 2).
std::size_t device_bound(const long long k, const std::size_t vertices) {
  return k < 0 ? std::max<std::size_t>(vertices, 2) : static_cast<std::size_t>(k);
}

std::string cuda_counts_case(const std::string& path, const int device) {
  namespace ts = cycle_enum::test_support;
  const test_case c = read_case(path);
  const cycle_enum::GraphView view0 = ts::view_from_edges(c.edges, c.n);
  const dyn::DirectedGraph g0 = dyn::build_directed_graph(view0);
  const cycle_enum::GraphView view1 = dyn::to_graph_view(dyn::apply_batch(g0, c.batch));
  std::string out;
  for (const long long k : {2LL, 3LL, 4LL, 5LL, 6LL, 7LL, -1LL}) {
    out += "k " + std::to_string(k) + "\n";
    cycle_enum::CycleHistogram before;
    std::string lines = cuda_static(view0, device_bound(k, view0.vertex_count()), device, "cuda",
                                    "_before", &before);
    // cuda_static names the automatic line "cuda_before".
    out += lines;
    out += histogram_text("cuda_after", cu::count_simple_cycles_johnson_work_queue(
                                            view1, device, device_bound(k, view1.vertex_count()),
                                            nullptr, cu::CudaWorkItems::Auto));
    if (k >= 0) {
      out += histogram_text("cuda_update",
                            dyn::update_static_histogram_cuda(g0, before, c.batch,
                                                              static_cast<std::size_t>(k), device));
    }
  }
  return out;
}

std::string cuda_count_file(const std::string& path, const long long k, const int device) {
  const cycle_enum::GraphView view = cycle_enum::read_graph_view(path);
  return cuda_static(view, device_bound(k, view.vertex_count()), device, "cuda", "", nullptr);
}

std::string cuda_update_file(const std::string& path, const std::size_t k,
                             const dyn::BatchParams& params, const int device) {
  const cycle_enum::GraphView view = cycle_enum::read_graph_view(path);
  const cycle_enum::CycleHistogram prior =
      cu::count_simple_cycles_johnson_work_queue(view, device, k, nullptr, cu::CudaWorkItems::Auto);
  const dyn::DirectedGraph g0 = dyn::build_directed_graph(view);
  const dyn::EdgeBatch batch = dyn::generate_batch(view, params);
  return "deletions " + std::to_string(batch.deletions.size()) + " insertions " +
         std::to_string(batch.insertions.size()) + "\n" + histogram_text("prior", prior) +
         histogram_text("cuda_update",
                        dyn::update_static_histogram_cuda(g0, prior, batch, k, device));
}
#endif

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3) {
    return usage();
  }
  const std::string command = argv[1];
  bool digest = argc > 3 && std::string(argv[argc - 1]) == "--digest";
  if (command == "parse") {
    emit(parse_text(cycle_enum::read_temporal_graph(argv[2])), digest);
    return 0;
  }
  if (command == "csr") {
    emit(csr_text(dyn::build_directed_graph(cycle_enum::read_graph_view(argv[2]))), digest);
    return 0;
  }
  if ((command == "generate" || command == "apply-generated") && argc >= 6) {
    const dyn::BatchParams params = batch_params(argv, argc, 3, digest);
    const cycle_enum::GraphView view = cycle_enum::read_graph_view(argv[2]);
    const dyn::EdgeBatch batch = dyn::generate_batch(view, params);
    if (command == "generate") {
      emit(batch_text(batch), digest);
      return 0;
    }
    const dyn::DirectedGraph g0 = dyn::build_directed_graph(view);
    const dyn::EdgeBatch prepared = dyn::prepare_batch(g0, batch);
    emit("prepared\n" + batch_text(prepared) + "after\n" + csr_text(dyn::apply_batch(g0, prepared)),
         digest);
    return 0;
  }
  if (command == "random-cases" && argc == 5) {
    random_cases(argv[2], std::atoi(argv[3]), std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  if (command == "apply" && argc == 3) {
    std::cout << apply_case(argv[2]);
    return 0;
  }
  if (command == "counts" && argc == 4) {
    std::cout << counts_case(argv[2], std::atoi(argv[3]));
    return 0;
  }
  if (command == "count-file" && argc >= 5) {
    bool omp_only = false;
    for (int i = 5; i < argc; ++i) {
      omp_only = omp_only || std::string(argv[i]) == "--omp-only";
    }
    emit(count_file(argv[2], std::atoll(argv[3]), std::atoi(argv[4]), omp_only), digest);
    return 0;
  }
  if (command == "update-file" && argc >= 8) {
    // <file> <k> <del> <ins> <seed> <threads> [<window>] [--digest]
    dyn::BatchParams params;
    params.num_deletions = std::strtoull(argv[4], nullptr, 10);
    params.num_insertions = std::strtoull(argv[5], nullptr, 10);
    params.seed = std::strtoull(argv[6], nullptr, 10);
    if (argc >= 9 && std::string(argv[8]) != "--digest") {
      params.locality_window = std::strtoull(argv[8], nullptr, 10);
    }
    emit(update_file(argv[2], std::strtoull(argv[3], nullptr, 10), params, std::atoi(argv[7])),
         digest);
    return 0;
  }
#if CYCLE_ENUM_CUDA_ENABLED
  if (command == "cuda-counts" && argc == 4) {
    std::cout << cuda_counts_case(argv[2], std::atoi(argv[3]));
    return 0;
  }
  if (command == "cuda-count-file" && argc >= 5) {
    emit(cuda_count_file(argv[2], std::atoll(argv[3]), std::atoi(argv[4])), digest);
    return 0;
  }
  if (command == "cuda-update-file" && argc >= 8) {
    // <file> <k> <del> <ins> <seed> <device> [<window>] [--digest]
    dyn::BatchParams params;
    params.num_deletions = std::strtoull(argv[4], nullptr, 10);
    params.num_insertions = std::strtoull(argv[5], nullptr, 10);
    params.seed = std::strtoull(argv[6], nullptr, 10);
    if (argc >= 9 && std::string(argv[8]) != "--digest") {
      params.locality_window = std::strtoull(argv[8], nullptr, 10);
    }
    emit(cuda_update_file(argv[2], std::strtoull(argv[3], nullptr, 10), params, std::atoi(argv[7])),
         digest);
    return 0;
  }
#endif
  return usage();
} catch (const std::exception& e) {
  std::cerr << "export_cycle_enum: " << e.what() << '\n';
  return 1;
}
