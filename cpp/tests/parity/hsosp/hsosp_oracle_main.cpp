// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_oracle_main.cpp
 * @brief dyng_hsosp_oracle: the go/no-go of M11a criterion 5 (ADR 0082) on H-SOSP dumps.
 *
 *     dyng_hsosp_oracle <dump.hsd[.zst]>...
 *
 * For every dump, dyng::testing's hyper_sssp oracle must reproduce MOSP_ESCHER's distances and
 * parents by id on every recorded update and recompute, and line_graph_delta its pair lists on
 * every batch (hsosp_dump::check_with_oracle). One JSON line per dump; exit status 1 when any
 * dump differs, 2 on a usage or read error. parity/hsosp_goldens.py oracle drives it over the
 * golden set.
 */
#include "parity/hsosp/hsosp_dump.hpp"

#include <cstdio>
#include <exception>
#include <string>

namespace {

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c == '\n' ? ' ' : c;
  }
  return out + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: dyng_hsosp_oracle <dump.hsd[.zst]>...\n");
    return 2;
  }
  int rc = 0;
  for (int i = 1; i < argc; ++i) {
    try {
      const auto recs = dyng::test::hsosp_dump::read_dump(argv[i]);
      const auto rep = dyng::test::hsosp_dump::check_with_oracle(recs);
      std::string problems = "[";
      for (std::size_t k = 0; k < rep.problems.size(); ++k)
        problems += (k ? ", " : "") + json_string(rep.problems[k]);
      problems += "]";
      std::printf(
          "{\"file\": %s, \"sessions\": %lld, \"sssp_checked\": %lld, \"distance_mismatches\": "
          "%lld, \"parent_mismatches\": %lld, \"batches_checked\": %lld, \"pair_mismatches\": "
          "%lld, \"not_checked\": %lld, \"ok\": %s, \"problems\": %s}\n",
          json_string(argv[i]).c_str(), static_cast<long long>(rep.sessions),
          static_cast<long long>(rep.sssp_checked),
          static_cast<long long>(rep.distance_mismatches),
          static_cast<long long>(rep.parent_mismatches),
          static_cast<long long>(rep.batches_checked), static_cast<long long>(rep.pair_mismatches),
          static_cast<long long>(rep.not_checked), rep.ok() ? "true" : "false", problems.c_str());
      std::fflush(stdout);
      if (!rep.ok()) rc = 1;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dyng_hsosp_oracle: %s: %s\n", argv[i], e.what());
      return 2;
    }
  }
  return rc;
}
