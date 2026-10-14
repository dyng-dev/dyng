// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_ids_main.cpp
 * @brief dyng_hypergraph_ids: container-level id parity of dyng::hypergraph against the batch
 *        records of ESCHER-GPU@abcf9b2's and MOSP_ESCHER@4b86159's CBST traces (M8b, criterion 4;
 *        ADR 0041, section 11). parity/compare.py hypergraph_ids drives it over the golden sets.
 *
 *     dyng_hypergraph_ids [--backend sequential|openmp|cuda] [--threads N]
 *                         [--mode by_id|by_content|both] [--no-model] [--no-integrity]
 *                         [--line-graph] [--apply-only] [--json OUT] [--tmp DIR] <trace>...
 *
 * --line-graph (ESCHER-GPU's traces): the by-id container keeps the line graph
 * (line_graph_policy::sorted_merge), its H2H is compared with ESCHER-GPU's after the build and
 * every batch (M8b criterion 7), and its device bytes with A's DynamicHypergraph::deviceBytes()
 * (the memory gate, criterion 8). --apply-only applies the batches and records the counters
 * without comparing rows (the memory and timing runs; implies --no-model --no-integrity).
 *
 * A trace is a `.cbt` file or a `.cbt.zst` (as dyng_cbst_replay reads them). Prints one line per
 * trace; --json writes a JSON array of the reports. Exit status: 0 every trace matched, 1 a
 * mismatch, 2 a usage or input error.
 */
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/hypergraph_ids.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: dyng_hypergraph_ids [--backend sequential|openmp|cuda] [--threads N] "
               "[--mode by_id|by_content|both] [--no-model] [--no-integrity] [--line-graph] "
               "[--apply-only] [--json OUT] [--tmp DIR] <trace>...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string backend = "sequential", json_path, tmp_dir;
  int threads = 0;
  dyng::test::hypergraph_ids::options opt;
  std::vector<std::string> traces;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--backend") {
      backend = value();
    } else if (a == "--threads") {
      threads = std::atoi(value().c_str());
    } else if (a == "--mode") {
      opt.mode = value();
      if (opt.mode != "by_id" && opt.mode != "by_content" && opt.mode != "both") return usage();
    } else if (a == "--no-model") {
      opt.model = false;
    } else if (a == "--no-integrity") {
      opt.integrity = false;
    } else if (a == "--line-graph") {
      opt.line_graph = true;
    } else if (a == "--apply-only") {
      opt.compare = false;
      opt.model = false;
      opt.integrity = false;
    } else if (a == "--json") {
      json_path = value();
    } else if (a == "--tmp") {
      tmp_dir = value();
    } else if (!a.empty() && a[0] == '-') {
      return usage();
    } else {
      traces.push_back(a);
    }
  }
  if (traces.empty()) return usage();
  dyng::resources res;
  try {
    if (backend == "sequential") {
      res = dyng::resources::sequential();
    } else if (backend == "openmp") {
      res = dyng::resources::openmp(threads);
    } else if (backend == "cuda") {
      res = dyng::resources::cuda(0);
    } else {
      return usage();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng_hypergraph_ids: %s\n", e.what());
    return 2;
  }
  bool all_ok = true;
  std::string json = "[";
  for (std::size_t t = 0; t < traces.size(); ++t) {
    try {
      dyng::test::cbst_compare::trace_source source(traces[t], tmp_dir);
      const auto rep = dyng::test::hypergraph_ids::replay(source.reader(), res, opt, traces[t]);
      std::printf(
          "%s %s %s: %s, %lld sessions, %lld batches, %lld ids, %lld rows, %lld mismatches%s\n",
          rep.ok() ? "ok" : "MISMATCH", rep.backend.c_str(), rep.mode.c_str(), traces[t].c_str(),
          static_cast<long long>(rep.sessions), static_cast<long long>(rep.batches),
          static_cast<long long>(rep.ids_compared), static_cast<long long>(rep.rows_compared),
          static_cast<long long>(rep.mismatch_count),
          rep.mismatches.empty() ? "" : (" (first: " + rep.mismatches[0] + ")").c_str());
      all_ok = all_ok && rep.ok();
      json += (t ? ",\n" : "\n") + rep.to_json();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dyng_hypergraph_ids: %s: %s\n", traces[t].c_str(), e.what());
      return 2;
    }
  }
  json += "\n]\n";
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    out << json;
    if (!out) {
      std::fprintf(stderr, "dyng_hypergraph_ids: cannot write %s\n", json_path.c_str());
      return 2;
    }
  }
  return all_ok ? 0 : 1;
}
