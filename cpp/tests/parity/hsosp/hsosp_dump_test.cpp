// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_dump_test.cpp
 * @brief The H-SOSP dump reader and the oracle go/no-go (M11a criteria 3 and 5; ADRs 0081 and
 *        0082) on the committed subset of the golden set hyper_sssp_mosp_escher
 *        (cpp/tests/data/goldens/hyper_sssp): it holds a vertex-kind batch and a forced fallback,
 *        dyng::testing's oracle reproduces MOSP_ESCHER's distances, parents and pairs on it, and
 *        a planted wrong distance, parent, pair, post-batch row or batch order fails the check.
 */
#include "parity/hsosp/hsosp_dump.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hd = dyng::test::hsosp_dump;

const std::string subset =
    std::string(DYNG_TEST_DATA_DIR) + "/goldens/hyper_sssp/test_hsosp_matches_dijkstra.hsd";

std::vector<hd::record> load() {
  return hd::read_dump(subset);
}

TEST(HsospDump, RefusesForeignAndTruncatedFiles) {
  EXPECT_THROW((void)hd::parse_dump("DYNGCBT1........"), hd::dump_error);
  EXPECT_THROW((void)hd::parse_dump("DYNGHSD1"), hd::dump_error);
  std::string bytes = "DYNGHSD1";
  bytes += std::string("\x02\0\0\0\0\0\0\0", 8);  // version 2
  EXPECT_THROW((void)hd::parse_dump(bytes), hd::dump_error);
  std::string good = "DYNGHSD1" + std::string("\x01\0\0\0\0\0\0\0", 8);
  EXPECT_TRUE(hd::parse_dump(good).empty());
  // A record whose payload runs past the end of the file.
  std::string trunc = good + std::string("\x09\0\0\0", 4) + std::string("\xff\0\0\0\0\0\0\0", 8);
  EXPECT_THROW((void)hd::parse_dump(trunc), hd::dump_error);
}

TEST(HsospDump, TheSubsetHasAVertexBatchAndAForcedFallback) {
  const auto recs = load();
  ASSERT_FALSE(recs.empty());
  EXPECT_EQ(recs.front().tag, "header");
  EXPECT_EQ(recs.back().tag, "end");
  bool vertex = false, fallback = false;
  std::int64_t sessions = 0;
  for (const auto& r : recs) {
    sessions += r.tag == "session";
    if (r.tag == "begin_batch" && r.str("phase") == "enter")
      vertex |= !r.ints("vtx_delete.he").empty() || !r.ints("vtx_insert.he").empty();
    if (r.tag == "sssp" && r.str("exception").empty())
      fallback |= r.i64("stats.fallback_recompute") == 1 && r.str("cfg.work_budget") == "0";
  }
  EXPECT_GE(sessions, 1);
  EXPECT_TRUE(vertex);
  EXPECT_TRUE(fallback);
}

TEST(HsospDump, TheOracleReproducesMospEscherOnTheSubset) {
  const auto rep = hd::check_with_oracle(load());
  EXPECT_TRUE(rep.ok()) << (rep.problems.empty() ? "" : rep.problems.front());
  EXPECT_GT(rep.sssp_checked, 0);
  EXPECT_GT(rep.batches_checked, 0);
  EXPECT_EQ(rep.not_checked, 0);
}

hd::record* nth(std::vector<hd::record>& recs, const std::string& tag, int n) {
  for (auto& r : recs) {
    if (r.tag == tag && (tag != "sssp" || r.str("exception").empty()) && n-- == 0) return &r;
  }
  return nullptr;
}

TEST(HsospDump, APlantedWrongDistanceOrParentFails) {
  auto recs = load();
  hd::record* s = nth(recs, "sssp", 1);
  ASSERT_NE(s, nullptr);
  s->fields.at("distances").longs[3] += 1;
  auto rep = hd::check_with_oracle(recs);
  EXPECT_FALSE(rep.ok());
  EXPECT_EQ(rep.distance_mismatches, 1);

  recs = load();
  s = nth(recs, "sssp", 1);
  auto& parents = s->fields.at("parents").ints;
  // A parent that is not the canonical one (any other id, or none).
  const auto it =
      std::find_if(parents.begin(), parents.end(), [](std::int32_t p) { return p >= 0; });
  ASSERT_NE(it, parents.end());
  *it = *it == 0 ? 1 : 0;
  rep = hd::check_with_oracle(recs);
  EXPECT_FALSE(rep.ok());
  EXPECT_EQ(rep.distance_mismatches, 0);
  EXPECT_EQ(rep.parent_mismatches, 1);
}

TEST(HsospDump, APlantedPairOrRowFails) {
  auto recs = load();
  hd::record* a = nth(recs, "apply_batch", 0);
  ASSERT_NE(a, nullptr);
  auto& del_a = a->fields.at("pairs.deleted.a").ints;
  auto& ins_a = a->fields.at("pairs.inserted.a").ints;
  ASSERT_FALSE(del_a.empty() && ins_a.empty());
  if (!del_a.empty()) {
    del_a.pop_back();
    a->fields.at("pairs.deleted.b").ints.pop_back();
  } else {
    ins_a.pop_back();
    a->fields.at("pairs.inserted.b").ints.pop_back();
  }
  EXPECT_GE(hd::check_with_oracle(recs).pair_mismatches, 1);

  // A post-batch row planted dead (a live hyperedge that shares a vertex with another): the
  // pairs and the distances no longer agree.
  recs = load();
  a = nth(recs, "apply_batch", 0);
  const auto& off = a->fields.at("rows.offsets").longs;
  const auto& val = a->fields.at("rows.values").ints;
  auto& alive = a->fields.at("rows.alive").ints;
  std::map<std::int32_t, int> holders;
  for (std::size_t k = 0; k + 1 < off.size(); ++k)
    if (alive[k])
      for (auto i = off[k]; i < off[k + 1]; ++i) ++holders[val[static_cast<std::size_t>(i)]];
  std::size_t row = off.size();
  for (std::size_t k = 0; k + 1 < off.size() && row == off.size(); ++k)
    for (auto i = off[k]; alive[k] && i < off[k + 1]; ++i)
      if (holders[val[static_cast<std::size_t>(i)]] >= 2) row = k;
  ASSERT_LT(row, off.size());
  alive[row] = 0;
  EXPECT_FALSE(hd::check_with_oracle(recs).ok());
}

TEST(HsospDump, AReorderedOrDroppedBatchFails) {
  auto recs = load();
  // Swap the first two apply_batch records of the first session that has two.
  std::vector<std::size_t> at;
  std::int64_t sid = -1;
  for (std::size_t k = 0; k < recs.size() && at.size() < 2; ++k) {
    if (recs[k].tag != "apply_batch") continue;
    if (sid == -1) sid = recs[k].i64("session");
    if (recs[k].i64("session") == sid) at.push_back(k);
  }
  ASSERT_EQ(at.size(), 2U);
  std::swap(recs[at[0]], recs[at[1]]);
  EXPECT_FALSE(hd::check_with_oracle(recs).ok());

  recs = load();
  recs.erase(recs.begin() + static_cast<std::ptrdiff_t>(at[0]));
  EXPECT_FALSE(hd::check_with_oracle(recs).ok());
}

}  // namespace
