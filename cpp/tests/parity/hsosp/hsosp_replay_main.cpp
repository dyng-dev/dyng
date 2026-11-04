// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_replay_main.cpp
 * @brief dyng_hsosp_replay: the container side of hyper_sssp's parity on MOSP_ESCHER's H-SOSP
 *        dumps (M11a criteria 8 and 10; hsosp_replay.hpp), and with --algorithm its result side
 *        (criterion 14; hsosp_results.hpp). parity/compare.py hyper_sssp --part delta and --part
 *        results drive it over the golden set.
 *
 *     dyng_hsosp_replay [--backend sequential|openmp|cuda] [--device D] [--threads N] [--no-oracle]
 *                       [--no-integrity] [--json OUT] <dump.hsd[.zst]>...
 *     dyng_hsosp_replay --algorithm by_id|by_id_public|by_key [--check-compute]
 *                       [--backend sequential|cuda] [--device D] [--json OUT] <dump.hsd[.zst]>...
 *
 * One line per dump; --json writes a JSON array of the reports. Exit status: 0 every dump
 * matched, 1 a mismatch, 2 a usage or input error.
 */
#include "parity/hsosp/hsosp_dump.hpp"
#include "parity/hsosp/hsosp_replay.hpp"
#include "parity/hsosp/hsosp_results.hpp"

#include <dyng/core/resources.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

int usage() {
  std::fprintf(
      stderr,
      "usage: dyng_hsosp_replay [--backend sequential|openmp|cuda] [--device D] [--threads N] "
      "[--no-oracle] [--no-integrity] [--json OUT] <dump.hsd[.zst]>...\n"
      "       dyng_hsosp_replay --algorithm by_id|by_id_public|by_key [--check-compute] "
      "[--backend sequential|cuda] [--device D] [--json OUT] <dump.hsd[.zst]>...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string backend = "sequential", json_path;
  int threads = 0;
  int device = 0;
  dyng::test::hsosp_replay::options opt;
  dyng::test::hsosp_results::options results;
  bool algorithm = false;
  std::vector<std::string> dumps;
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
    } else if (a == "--device") {
      device = std::atoi(value().c_str());
    } else if (a == "--no-oracle") {
      opt.oracle = false;
    } else if (a == "--no-integrity") {
      opt.integrity = false;
    } else if (a == "--json") {
      json_path = value();
    } else if (a == "--algorithm") {
      algorithm = true;
      const std::string m = value();
      if (m == "by_id") {
        results.mode = dyng::test::hsosp_results::mode::by_id;
      } else if (m == "by_id_public") {
        results.mode = dyng::test::hsosp_results::mode::by_id_public;
      } else if (m == "by_key") {
        results.mode = dyng::test::hsosp_results::mode::by_key;
      } else {
        return usage();
      }
    } else if (a == "--check-compute") {
      results.check_compute = true;
    } else if (!a.empty() && a[0] == '-') {
      return usage();
    } else {
      dumps.push_back(a);
    }
  }
  if (dumps.empty()) return usage();
  dyng::resources res;
  try {
    if (backend == "sequential") {
      res = dyng::resources::sequential();
    } else if (backend == "openmp") {
      res = dyng::resources::openmp(threads);
    } else if (backend == "cuda") {
      res = dyng::resources::cuda(device, {}, threads);
    } else {
      return usage();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng_hsosp_replay: %s\n", e.what());
    return 2;
  }
  bool all_ok = true;
  std::string json = "[";
  for (std::size_t d = 0; d < dumps.size(); ++d) {
    try {
      const auto records = dyng::test::hsosp_dump::read_dump(dumps[d]);
      if (algorithm) {
#if defined(DYNG_HSOSP_RESULTS) && DYNG_HSOSP_RESULTS
        const auto rep = dyng::test::hsosp_results::replay(records, res, results, dumps[d]);
        std::printf(
            "%s %s %s: %lld sessions, %lld batches, %lld updates, %lld recomputes (%lld by "
            "digest), %lld ids, fallbacks %lld (B %lld), %lld mismatches%s\n",
            rep.ok() ? "ok" : "MISMATCH", rep.mode.c_str(), dumps[d].c_str(),
            static_cast<long long>(rep.sessions), static_cast<long long>(rep.batches),
            static_cast<long long>(rep.updates), static_cast<long long>(rep.recomputes),
            static_cast<long long>(rep.digest_calls), static_cast<long long>(rep.ids_compared),
            static_cast<long long>(rep.dyng_fallbacks), static_cast<long long>(rep.b_fallbacks),
            static_cast<long long>(rep.mismatch_count),
            rep.mismatches.empty() ? "" : (" (first: " + rep.mismatches[0] + ")").c_str());
        std::fflush(stdout);
        all_ok = all_ok && rep.ok();
        json += (d ? ",\n" : "\n") + rep.to_json();
        continue;
#else
        std::fprintf(stderr,
                     "dyng_hsosp_replay: --algorithm needs hyper_sssp, which this build "
                     "leaves out (DYNG_ALGORITHMS)\n");
        return 2;
#endif
      }
      const auto rep = dyng::test::hsosp_replay::replay(records, res, opt, dumps[d]);
      std::printf(
          "%s %s: %lld sessions, %lld batches (%lld with incidence edits), %lld ids, "
          "%lld + %lld pairs, %lld rows, %lld mismatches%s\n",
          rep.ok() ? "ok" : "MISMATCH", dumps[d].c_str(), static_cast<long long>(rep.sessions),
          static_cast<long long>(rep.batches), static_cast<long long>(rep.vertex_batches),
          static_cast<long long>(rep.ids_compared), static_cast<long long>(rep.deleted_pairs),
          static_cast<long long>(rep.inserted_pairs), static_cast<long long>(rep.rows_compared),
          static_cast<long long>(rep.mismatch_count),
          rep.mismatches.empty() ? "" : (" (first: " + rep.mismatches[0] + ")").c_str());
      std::fflush(stdout);
      all_ok = all_ok && rep.ok();
      json += (d ? ",\n" : "\n") + rep.to_json();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dyng_hsosp_replay: %s: %s\n", dumps[d].c_str(), e.what());
      return 2;
    }
  }
  json += "\n]\n";
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    out << json;
    if (!out) {
      std::fprintf(stderr, "dyng_hsosp_replay: cannot write %s\n", json_path.c_str());
      return 2;
    }
  }
  return all_ok ? 0 : 1;
}
