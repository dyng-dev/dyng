// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file escher_dump_test.cpp
 * @brief ESCHER-GPU's batch dumps on the parity side: the decoder on the committed traces, A's
 *        command line, and the by-content resolver's deterministic rule.
 */
#include "parity/escher/escher_dump.hpp"

#include "parity/escher/cbst_trace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace ed = dyng::test::escher_dump;
namespace tr = dyng::test::cbst_trace;

fs::path trace(const std::string& relative) {
  return fs::path(DYNG_TEST_DATA_DIR) / "goldens" / relative;
}

TEST(EscherDump, DecodesTheBuildAndTheBatches) {
  auto in = tr::reader::open(trace("hypergraph_batches/escher_gpu_016-main.cbt").string());
  ed::decoder d;
  tr::record rec;
  int builds = 0;
  std::vector<ed::batch_record> batches;
  while (in->next(rec)) {
    switch (d.feed(rec, *in)) {
      case ed::decoder::event::build:
        ++builds;
        EXPECT_EQ(d.build().num_vertices, 2000);
        EXPECT_EQ(d.build().loaded.size(), 3000u);
        break;
      case ed::decoder::event::batch:
        batches.push_back(d.batch());
        break;
      case ed::decoder::event::exception:
        ADD_FAILURE() << d.exception();
        break;
      default:
        break;
    }
  }
  EXPECT_EQ(d.repo(), "ESCHER-GPU");
  EXPECT_EQ(builds, 1);
  ASSERT_EQ(batches.size(), 3u);
  for (const auto& b : batches) {
    EXPECT_EQ(b.deleted_ids.size(), 10u);  // --batch-size 20: 10 deletions, 10 insertions
    EXPECT_EQ(b.deleted_rows.size(), 10u);
    EXPECT_EQ(b.inserted_rows.size(), 10u);
    EXPECT_EQ(b.inserted_ids.size(), 10u);
    EXPECT_TRUE(std::is_sorted(b.deleted_ids.begin(), b.deleted_ids.end()));
    for (const auto& r : b.inserted_rows) {
      for (const std::int32_t v : r) EXPECT_GE(v, 0);  // 0-based
    }
  }
  const ed::escher_options o = ed::escher_options::parse(d.argv());
  EXPECT_TRUE(o.from_file);
  EXPECT_TRUE(o.sample_inserts);  // the default with --input
  EXPECT_TRUE(o.composition_set);
  EXPECT_EQ(o.deletions, 10);
  EXPECT_EQ(o.insertions, 10);
  EXPECT_EQ(o.seed, 3u);
  EXPECT_EQ(o.batches, 3);
}

TEST(EscherDump, ParsesEschersCommandLine) {
  const auto o = ed::escher_options::parse({"main", "8", "5", "1", "100", "4096"});
  EXPECT_FALSE(o.from_file);
  EXPECT_EQ(o.num_hyperedges, 8);
  EXPECT_EQ(o.max_vertices_per_edge, 5);
  EXPECT_EQ(o.min_vertex_id, 1);
  EXPECT_EQ(o.max_vertex_id, 100);
  EXPECT_EQ(o.payload_capacity, 4096);
  EXPECT_FALSE(o.composition_set);
  EXPECT_FALSE(o.sample_inserts);
  EXPECT_EQ(o.seed, 1u);
  const auto p = ed::escher_options::parse(
      {"main", "3000", "8", "1", "2000", "--batch-size", "401", "--insert-mode", "sample"});
  EXPECT_EQ(p.deletions, 200);
  EXPECT_EQ(p.insertions, 201);
  EXPECT_TRUE(p.sample_inserts);
  EXPECT_THROW((void)ed::escher_options::parse({"main", "8", "5"}), tr::trace_error);
  EXPECT_THROW((void)ed::escher_options::parse({"main", "--input", "x", "--frob"}),
               tr::trace_error);
}

TEST(EscherDump, TheResolverTakesTheSmallestLiveIdNotTakenYet) {
  ed::by_content_resolver r;
  // Ids 0 and 3 share a member set; id 2 is dead.
  r.reset({{1, 2}, {5}, {}, {2, 1}, {7, 8, 9}});
  EXPECT_EQ(r.size(), 4u);
  std::vector<bool> ambiguous;
  const auto ids = r.resolve({{2, 1, 1}, {1, 2}, {1, 2}, {9, 8, 7}, {4}}, &ambiguous);
  EXPECT_EQ(ids, (std::vector<std::int64_t>{0, 3, -1, 4, -1}));
  EXPECT_EQ(ambiguous, (std::vector<bool>{true, true, true, false, false}));
  // A commit: 0 and 4 deleted, {1, 2} inserted at id 2.
  r.commit({0, 4}, {{2, 1}}, {2});
  EXPECT_EQ(r.resolve({{1, 2}, {1, 2}}), (std::vector<std::int64_t>{2, 3}));
  EXPECT_EQ(r.resolve({{7, 8, 9}}), (std::vector<std::int64_t>{-1}));
  EXPECT_EQ(r.size(), 3u);
}

}  // namespace
