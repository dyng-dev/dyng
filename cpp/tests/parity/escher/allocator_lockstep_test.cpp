// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocator_lockstep_test.cpp
 * @brief The host id allocator in lockstep with MOSP_ESCHER@4b86159's recorded CBST trees and the
 *        host model (M11a criterion 8; ADR 0085): the committed traces (hosted CI) and, when the
 *        goldens are in $DYNG_SCRATCH, make test's H-SOSP traces with their vertex-kind batches.
 */
#include "parity/escher/allocator_lockstep.hpp"

#include "parity/escher/cbst_trace.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace al = dyng::test::allocator_lockstep;

fs::path committed() {
  return fs::path(DYNG_TEST_DATA_DIR) / "goldens" / "cbst_trace";
}

fs::path scratch() {
  if (const char* s = std::getenv("DYNG_SCRATCH")) return s;
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "/") / "Projects" / "dyng-work";
}

al::report run(const fs::path& path, const std::string& trees) {
  dyng::test::cbst_trace::trace_source source(path.string(), fs::temp_directory_path().string());
  al::options opt;
  opt.trees = trees;
  return al::run(source.reader(), opt, path.string());
}

// The targeted scenarios drive CBSTOperations directly: fills that chain rows, unfills of 40 and
// 1,100 values, compaction inside fill and before a surplus insert, an explicit compact().
TEST(AllocatorLockstep, TheCommittedTargetedTraceOfMospEscher) {
  const al::report r = run(committed() / "mosp_escher_targeted" / "targeted_small.cbt", "all");
  EXPECT_TRUE(r.ok()) << r.to_json();
  EXPECT_GT(r.trees, 0);
  EXPECT_GT(r.fills, 0);
  EXPECT_GT(r.unfills, 0);
  EXPECT_GT(r.multi_segment_unfills, 0);
  EXPECT_GT(r.compactions, 0);
  EXPECT_EQ(r.unreplayable, 0);
}

TEST(AllocatorLockstep, TheCommittedTestCbstOpsTraces) {
  for (const char* name :
       {"004-test_cbst_ops.cbt", "005-test_cbst_ops.cbt", "006-test_cbst_ops.cbt",
        "007-test_cbst_ops.cbt", "008-test_cbst_ops.cbt"}) {
    SCOPED_TRACE(name);
    const al::report r = run(committed() / "mosp_escher_test" / name, "all");
    EXPECT_TRUE(r.ok()) << r.to_json();
    EXPECT_GT(r.states, 0);
  }
}

// make test's H-SOSP sessions: the H2V tree of every DynamicHypergraph through its hyperedge- and
// vertex-kind batches (from the goldens; skipped without them).
TEST(AllocatorLockstep, MakeTestsHsospTracesFromTheGoldens) {
  const fs::path set = scratch() / "goldens" / "cbst_trace" / "mosp_escher_test";
  const std::vector<std::string> cases = {"014-test_h2h_delta", "015-test_hsosp_matches_dijkstra"};
  if (!fs::is_directory(set)) GTEST_SKIP() << "no goldens under " << set;
  for (const std::string& c : cases) {
    SCOPED_TRACE(c);
    const fs::path trace = set / c / "trace.cbt.zst";
    if (!fs::is_regular_file(trace)) GTEST_SKIP() << "no " << trace;
    const al::report r = run(trace, "h2v");
    EXPECT_TRUE(r.ok()) << r.to_json();
    EXPECT_GT(r.inserts, 0);
    if (c == "015-test_hsosp_matches_dijkstra") {
      EXPECT_GT(r.fills, 0);
      EXPECT_GT(r.unfills, 0);
    }
  }
}

}  // namespace
