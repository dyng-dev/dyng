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
#include "cycle_enum/sequential/bruteforce.hpp"
#include "support/cycle_oracles.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace dyn = cycle_enum::dynamic;

int usage() {
  std::cerr << "usage: export_cycle_enum parse|csr|generate|apply-generated|random-cases|apply "
               "...\n";
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
  return usage();
} catch (const std::exception& e) {
  std::cerr << "export_cycle_enum: " << e.what() << '\n';
  return 1;
}
