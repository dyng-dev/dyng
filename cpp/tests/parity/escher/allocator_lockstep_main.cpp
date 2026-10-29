// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocator_lockstep_main.cpp
 * @brief dyng_allocator_lockstep: the host id allocator in lockstep with the CBST trees of
 *        MOSP_ESCHER@4b86159's traces and with the host model (M11a criterion 8; ADR 0085).
 *        parity/hypergraph_ids.py allocator drives it over the golden sets.
 *
 *     dyng_allocator_lockstep [--trees h2v|all] [--json OUT] [--tmp DIR] <trace>...
 *
 * A trace is a `.cbt` file or a `.cbt.zst`. Prints one line per trace; --json writes a JSON
 * array of the reports. Exit status: 0 every trace matched, 1 a mismatch, 2 a usage or input
 * error.
 */
#include "parity/escher/allocator_lockstep.hpp"
#include "parity/escher/cbst_trace.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: dyng_allocator_lockstep [--trees h2v|all] [--json OUT] [--tmp DIR] "
               "<trace>...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  dyng::test::allocator_lockstep::options opt;
  std::string json_path, tmp_dir;
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
    if (a == "--trees") {
      opt.trees = value();
      if (opt.trees != "h2v" && opt.trees != "all") return usage();
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
  bool all_ok = true;
  std::string json = "[";
  for (std::size_t t = 0; t < traces.size(); ++t) {
    try {
      dyng::test::cbst_trace::trace_source source(traces[t], tmp_dir);
      const auto rep = dyng::test::allocator_lockstep::run(source.reader(), opt, traces[t]);
      std::printf(
          "%s %s: %lld trees, %lld ops, %lld states, %lld fills, %lld unfills (%lld items on "
          "rows with an overflow segment), %lld compactions, %lld mismatches%s\n",
          rep.ok() ? "ok" : "MISMATCH", traces[t].c_str(), static_cast<long long>(rep.trees),
          static_cast<long long>(rep.ops), static_cast<long long>(rep.states),
          static_cast<long long>(rep.fills), static_cast<long long>(rep.unfills),
          static_cast<long long>(rep.multi_segment_unfills),
          static_cast<long long>(rep.compactions), static_cast<long long>(rep.mismatch_count),
          rep.mismatches.empty() ? "" : (" (first: " + rep.mismatches[0] + ")").c_str());
      all_ok = all_ok && rep.ok();
      json += (t ? ",\n" : "\n") + rep.to_json();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dyng_allocator_lockstep: %s: %s\n", traces[t].c_str(), e.what());
      return 2;
    }
  }
  json += "\n]\n";
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    out << json;
    if (!out) {
      std::fprintf(stderr, "dyng_allocator_lockstep: cannot write %s\n", json_path.c_str());
      return 2;
    }
  }
  return all_ok ? 0 : 1;
}
