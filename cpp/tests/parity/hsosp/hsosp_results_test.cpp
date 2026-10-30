// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_results_test.cpp
 * @brief hyper_sssp's result-side replay (M11a criterion 14; ADR 0091) on the committed subset of
 *        the golden set hyper_sssp_mosp_escher: every session replayed whole through hyper_sssp
 *        equals MOSP_ESCHER's updates and recomputes in the three modes (by_id, by_id_public,
 *        by_key), the fallback decisions are tabulated (the subset forces one), update == compute
 *        holds, and the comparator catches a planted wrong distance or parent, a reordered batch
 *        and a dropped batch. Compiled for the host (sequential) and, with DYNG_TEST_CUDA=1, for
 *        cuda.
 */
#include "parity/hsosp/hsosp_results.hpp"

#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace hd = dyng::test::hsosp_dump;
namespace hr = dyng::test::hsosp_results;

const std::string subset =
    std::string(DYNG_TEST_DATA_DIR) + "/goldens/hyper_sssp/test_hsosp_matches_dijkstra.hsd";

dyng::resources backend() {
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
  return dyng::resources::cuda(0);
#else
  return dyng::resources::sequential();
#endif
}

hr::report replay(const std::vector<hd::record>& recs, hr::mode m = hr::mode::by_id,
                  bool check_compute = false) {
  hr::options opt;
  opt.mode = m;
  opt.check_compute = check_compute;
  return hr::replay(recs, backend(), opt, "subset");
}

TEST(HsospResults, TheSubsetEqualsMospEscherInEveryMode) {
  const auto recs = hd::read_dump(subset);
  for (const hr::mode m : {hr::mode::by_id, hr::mode::by_id_public, hr::mode::by_key}) {
    const hr::report rep = replay(recs, m, true);
    SCOPED_TRACE(rep.mode);
    EXPECT_TRUE(rep.ok()) << (rep.mismatches.empty() ? "" : rep.mismatches.front());
    EXPECT_EQ(rep.stopped_sessions, 0);
    EXPECT_GT(rep.updates, 0);
    EXPECT_GT(rep.recomputes, 0);
    EXPECT_GT(rep.ids_compared, 0);
    EXPECT_EQ(rep.compute_checked, rep.updates - rep.not_checked);
    EXPECT_EQ(rep.not_checked, 0);
    EXPECT_GT(rep.b_fallbacks, 0);  // the subset forces a fallback (workBudget 0)
    if (m != hr::mode::by_id) {
      EXPECT_GT(rep.canonical_checked, 0);
    }
  }
}

TEST(HsospResults, TheFallbackDecisionsAreTabulated) {
  const hr::report rep = replay(hd::read_dump(subset));
  EXPECT_EQ(rep.fallback_equal + rep.fallback_differ, rep.updates - rep.not_checked);
  EXPECT_EQ(rep.dyng_fallbacks, rep.b_fallbacks);  // workBudget 0: deterministic on both sides
  ASSERT_FALSE(rep.calls.empty());
  for (const hr::call_digest& c : rep.calls) EXPECT_EQ(c.distances.size(), 64U);
}

/// The index of the n-th record with `tag` (and `kind`, for sssp records), or recs.size().
std::size_t nth(const std::vector<hd::record>& recs, const std::string& tag, int n,
                const std::string& kind = "") {
  for (std::size_t i = 0; i < recs.size(); ++i) {
    if (recs[i].tag != tag) continue;
    if (!kind.empty() && recs[i].str("kind") != kind) continue;
    if (recs[i].has("exception") && !recs[i].str("exception").empty()) continue;
    if (n-- == 0) return i;
  }
  return recs.size();
}

void expect_caught(const std::vector<hd::record>& recs, const std::string& what,
                   hr::mode m = hr::mode::by_id) {
  const hr::report rep = replay(recs, m);
  EXPECT_FALSE(rep.ok()) << what;
  EXPECT_GE(rep.mismatch_count, 1) << what;
}

TEST(HsospResults, APlantedDistanceOrParentIsCaught) {
  auto recs = hd::read_dump(subset);
  std::size_t u = nth(recs, "sssp", 1, "update");
  ASSERT_LT(u, recs.size());
  auto& d = recs[u].fields.at("distances").longs;
  const auto finite = std::find_if(d.begin() + 1, d.end(),
                                   [](std::int64_t x) { return x > 0 && x < INT64_MAX / 4; });
  ASSERT_NE(finite, d.end());
  *finite += 1;
  expect_caught(recs, "a wrong distance of an update");
  expect_caught(recs, "a wrong distance (by_id_public)", hr::mode::by_id_public);
  expect_caught(recs, "a wrong distance (by_key)", hr::mode::by_key);

  recs = hd::read_dump(subset);
  u = nth(recs, "sssp", 0, "recompute");
  ASSERT_LT(u, recs.size());
  auto& p = recs[u].fields.at("parents").ints;
  const auto with_parent = std::find_if(p.begin(), p.end(), [](std::int32_t x) { return x >= 0; });
  ASSERT_NE(with_parent, p.end());
  *with_parent = *with_parent == 0 ? 1 : 0;
  expect_caught(recs, "a wrong parent of a recompute");
}

/// The records of batch k of the first session: [begin_batch enter, ..., the last sssp record
/// before the next batch).
std::pair<std::size_t, std::size_t> batch_group(const std::vector<hd::record>& recs, int k) {
  const std::size_t first = nth(recs, "begin_batch", 2 * k);
  std::size_t last = first + 1;
  while (last < recs.size() && recs[last].tag != "session" &&
         !(recs[last].tag == "begin_batch" && recs[last].str("phase") == "enter"))
    ++last;
  return {first, last};
}

TEST(HsospResults, AReorderedOrDroppedBatchIsCaught) {
  auto recs = hd::read_dump(subset);
  auto [a0, a1] = batch_group(recs, 0);
  auto [b0, b1] = batch_group(recs, 1);
  ASSERT_EQ(a1, b0);
  ASSERT_LT(b0, recs.size());
  ASSERT_EQ(recs[a0].i64("session"), recs[b0].i64("session"));
  std::vector<hd::record> swapped(recs.begin(), recs.begin() + static_cast<std::ptrdiff_t>(a0));
  // The two batches' operations swapped, each keeping the other's B results.
  std::vector<hd::record> first(recs.begin() + static_cast<std::ptrdiff_t>(a0),
                                recs.begin() + static_cast<std::ptrdiff_t>(a1));
  std::vector<hd::record> second(recs.begin() + static_cast<std::ptrdiff_t>(b0),
                                 recs.begin() + static_cast<std::ptrdiff_t>(b1));
  const auto ops_end = [](const std::vector<hd::record>& g) {
    std::size_t i = 0;
    while (i < g.size() && g[i].tag != "sssp") ++i;
    return static_cast<std::ptrdiff_t>(i);
  };
  swapped.insert(swapped.end(), second.begin(), second.begin() + ops_end(second));
  swapped.insert(swapped.end(), first.begin() + ops_end(first), first.end());
  swapped.insert(swapped.end(), first.begin(), first.begin() + ops_end(first));
  swapped.insert(swapped.end(), second.begin() + ops_end(second), second.end());
  swapped.insert(swapped.end(), recs.begin() + static_cast<std::ptrdiff_t>(b1), recs.end());
  expect_caught(swapped, "two batches in the other order");

  // A dropped batch: its operations removed, B's results after it kept.
  std::vector<hd::record> dropped(recs.begin(), recs.begin() + static_cast<std::ptrdiff_t>(a0));
  dropped.insert(dropped.end(), first.begin() + ops_end(first), first.end());
  dropped.insert(dropped.end(), recs.begin() + static_cast<std::ptrdiff_t>(a1), recs.end());
  expect_caught(dropped, "a dropped batch");
}

}  // namespace
