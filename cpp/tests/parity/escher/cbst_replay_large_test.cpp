// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_replay_large_test.cpp
 * @brief The host model replays every CBST trace golden set in $DYNG_SCRATCH/goldens/cbst_trace
 *        (M8a; ADR 0036 and 0037) with 100% agreement; skipped when the sets are absent.
 *
 * Labels parity, cbst and slow (not cpu: the sets live outside the repository). Every case of
 * every set present is replayed under its original's preset with the field set
 * $DYNG_CBST_FIELDS (semantic, offsets or exact; default exact). Operations that cannot be
 * replayed are allowed only where the set is documented to make them so: the batches-mode set
 * escher_gpu_syn1m (no state digests) and the H2H rebuilds of escher_gpu_coauth50k (hash-only
 * inputs, parity/goldens.toml). parity/compare.py cbst_trace runs the same replays with a JSON
 * record per set.
 */
#include "parity/escher/cbst_compare.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

namespace fs = std::filesystem;

fs::path scratch() {
  if (const char* s = std::getenv("DYNG_SCRATCH")) return s;
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "/") / "Projects" / "dyng-work";
}

struct case_result {
  std::string set, name;
  replay_report report;
  std::string error;
};

TEST(CbstLargeSets, TheHostModelReplaysEverySet) {
  const fs::path root = scratch() / "goldens" / "cbst_trace";
  if (!fs::is_directory(root)) {
    GTEST_SKIP() << "the CBST trace golden sets are not under " << root
                 << " (export them with parity/export_goldens.py cbst_trace all)";
  }
  field_set fields = field_set::exact;
  if (const char* f = std::getenv("DYNG_CBST_FIELDS")) {
    ASSERT_TRUE(parse_field_set(f, fields)) << f;
  }
  std::vector<case_result> cases;
  for (const auto& set : fs::directory_iterator(root)) {
    if (!set.is_directory()) continue;
    for (const auto& c : fs::directory_iterator(set.path()))
      if (fs::is_regular_file(c.path() / "trace.cbt.zst"))
        cases.push_back({set.path().filename().string(), c.path().filename().string(), {}, {}});
  }
  if (cases.empty()) {
    GTEST_SKIP() << "no traces under " << root;
  }
  std::sort(cases.begin(), cases.end(), [](const case_result& a, const case_result& b) {
    return a.set + a.name < b.set + b.name;
  });
  const fs::path tmp = scratch() / "runs";
  fs::create_directories(tmp);
  std::atomic<std::size_t> next{0};
  const unsigned threads = std::max(1U, std::min(8U, std::thread::hardware_concurrency() / 4));
  std::vector<std::thread> pool;
  for (unsigned w = 0; w < threads; ++w)
    pool.emplace_back([&] {
      for (std::size_t i = next++; i < cases.size(); i = next++) {
        case_result& c = cases[i];
        try {
          trace_source source((root / c.set / c.name / "trace.cbt.zst").string(), tmp.string());
          options opt;
          opt.fields = fields;
          c.report = replay(source.reader(), host_model_factory(""), opt, c.set + "/" + c.name);
        } catch (const std::exception& e) {
          c.error = e.what();
        }
      }
    });
  for (auto& t : pool) t.join();
  std::int64_t ops = 0, compared = 0;
  for (const case_result& c : cases) {
    SCOPED_TRACE(c.set + "/" + c.name);
    ASSERT_TRUE(c.error.empty()) << c.error;
    const replay_report& r = c.report;
    ops += r.ops;
    compared += r.compared;
    EXPECT_TRUE(r.ok()) << r.mismatch_count << " mismatches; first: "
                        << (r.mismatches.empty()
                                ? std::string()
                                : r.mismatches.front().op + " " + r.mismatches.front().field +
                                      ": trace " + r.mismatches.front().expected + ", model " +
                                      r.mismatches.front().actual);
    for (const auto& [id, t] : r.trees) {
      if (t.unreplayable == 0) continue;
      const bool batches = c.set == "escher_gpu_syn1m";
      const bool coauth_h2h = c.set == "escher_gpu_coauth50k" && t.dataset == "H2H" &&
                              t.unreplayable_reason.find("hash-only inputs") != std::string::npos;
      EXPECT_TRUE(batches || coauth_h2h)
          << "tree " << id << " (" << t.dataset << "): " << t.unreplayable
          << " ops unreplayable: " << t.unreplayable_reason;
    }
  }
  std::printf("[ cbst ] %zu traces, %lld ops, %lld compared (%s)\n", cases.size(),
              static_cast<long long>(ops), static_cast<long long>(compared), to_string(fields));
  EXPECT_GT(compared, 0);
}

}  // namespace
}  // namespace dyng::test::cbst_compare
