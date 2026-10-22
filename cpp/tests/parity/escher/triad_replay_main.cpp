// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_replay_main.cpp
 * @brief dyng_triad_replay: one case of ESCHER-GPU@abcf9b2's golden set triad_count_escher_gpu
 *        replayed through dyng::hypergraph with a triad counter after the build and every batch
 *        (M9a step 2; ADR 0050). parity/compare.py triad_count drives it and compares its report
 *        with the goldens.
 *
 *     dyng_triad_replay --hg FILE [--vertices N] [--capacity C]
 *                       [--backend sequential|openmp|cuda] [--threads N]
 *                       [--mode by_content|by_key|both] [--counter oracle|algorithm|local_patterns]
 *                       [--patterns h_motif,stathyper_type1,...] [--oracle-threads N]
 *                       [--no-integrity] [--json OUT] [batch.dgb ...]
 *
 * --vertices is A's vertex count (group.json `vertices`), --capacity the H2V payload capacity
 * (group.json `payload_capacity`; 0 = A's formula). With no batch file it reports the full count
 * only (the single-triad check of the oracle against A, M9a criterion 4). Exit status: 0 the
 * replay ran (the comparison is the caller's), 2 a usage or input error, or the counter is not
 * available (`--counter algorithm` before triad_count exists; `--counter local_patterns`
 *        without CUDA).
 */
#include "parity/escher/triad_replay.hpp"

#include <dyng/core/resources.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int usage() {
  std::fprintf(
      stderr,
      "usage: dyng_triad_replay --hg FILE [--vertices N] [--capacity C] "
      "[--backend sequential|openmp|cuda] [--threads N] "
      "[--mode by_content|by_key|both] [--counter oracle|algorithm|local_patterns] "
      "[--patterns P,...] [--oracle-threads N] [--no-integrity] [--no-recount] [--json OUT] "
      "[batch.dgb ...]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  namespace trp = dyng::test::triad_replay;
  std::string backend = "sequential", json_path, hg;
  int threads = 0;
  trp::options opt;
  std::vector<std::string> batches;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto value = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--hg") {
        hg = value();
      } else if (a == "--vertices") {
        opt.num_vertices = static_cast<std::int32_t>(std::stol(value()));
      } else if (a == "--capacity") {
        opt.payload_capacity = std::stoll(value());
      } else if (a == "--backend") {
        backend = value();
      } else if (a == "--threads") {
        threads = std::stoi(value());
      } else if (a == "--mode") {
        opt.mode = value();
      } else if (a == "--counter") {
        opt.counter = value();
      } else if (a == "--patterns") {
        opt.patterns.clear();
        std::stringstream list(value());
        for (std::string p; std::getline(list, p, ',');) opt.patterns.push_back(trp::pattern_of(p));
      } else if (a == "--oracle-threads") {
        opt.oracle_threads = std::stoi(value());
      } else if (a == "--no-integrity") {
        opt.integrity = false;
      } else if (a == "--no-recount") {
        opt.recount = false;
      } else if (a == "--json") {
        json_path = value();
      } else if (!a.empty() && a[0] == '-') {
        return usage();
      } else {
        batches.push_back(a);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng_triad_replay: %s\n", e.what());
    return usage();
  }
  if (hg.empty() || opt.patterns.empty()) return usage();
  try {
    dyng::resources res;
    if (backend == "sequential") {
      res = dyng::resources::sequential();
    } else if (backend == "openmp") {
      res = dyng::resources::openmp(threads);
    } else if (backend == "cuda") {
      res = dyng::resources::cuda(0);
    } else {
      return usage();
    }
    const trp::report rep = trp::replay(res, hg, batches, opt);
    const std::string json = rep.to_json();
    if (!json_path.empty()) {
      std::ofstream out(json_path);
      out << json;
      if (!out) {
        std::fprintf(stderr, "dyng_triad_replay: cannot write %s\n", json_path.c_str());
        return 2;
      }
    } else {
      std::fputs(json.c_str(), stdout);
    }
    std::fprintf(stderr, "dyng_triad_replay: %s %s, %zu batch file(s), %.1f s\n",
                 rep.backend.c_str(), opt.mode.c_str(), batches.size(), rep.seconds);
  } catch (const trp::not_available& e) {
    std::fprintf(stderr, "dyng_triad_replay: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng_triad_replay: %s: %s\n", hg.c_str(), e.what());
    return 2;
  }
  return 0;
}
