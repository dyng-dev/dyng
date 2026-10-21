// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_limits_test.cpp
 * @brief triad_count at the 64-bit limit (ADR 0057): counts between 2^63 and 2^64 - 1 are exact,
 *        a count past 2^64 - 1 throws capacity_error, and an update that changes a bin by 2^63 or
 *        more throws capacity_error, the same on every backend; the oracle is exact there too and
 *        refuses what does not fit (std::overflow_error).
 *
 * The inputs are a few hyperedges of millions of members (C(|e|, 3) passes 2^63 - 1 from |e| =
 * 2,097,152 as ESCHER-GPU computes it, n (n - 1) (n - 2) / 6 in signed 64-bit; x * y * z of type 3
 * passes 2^64 - 1 from x = y = z = 2,642,246), so the host executable is labeled parity and slow
 * (ci/check.sh's parity step, a Release build) and the CUDA one gpu (ci/gpu_local.sh).
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/triad_reference.hpp>
#include <dyng/triad_count.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace tc = dyng::triad_count;
namespace tr = dyng::testing;
using dyng::backend;
using dyng::resources;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using batch_t = dyng::hyperedge_batch<std::int32_t, dyng::unweighted>;
using row = std::vector<std::int32_t>;
using rows = std::vector<row>;

/// The vertices [lo, hi).
row range(std::int32_t lo, std::int32_t hi) {
  row r(static_cast<std::size_t>(hi - lo));
  std::iota(r.begin(), r.end(), lo);
  return r;
}

/// The union of two disjoint sorted rows (sorted).
row join(const row& a, const row& b) {
  row r = a;
  r.insert(r.end(), b.begin(), b.end());
  return r;
}

hgraph make(const resources& res, std::int32_t n, const rows& rs, bool line_graph) {
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = n;
  for (const auto& r : rs) list.add_hyperedge(dyng::host_view(r));
  dyng::hypergraph_properties props;
  props.line_graph =
      line_graph ? dyng::line_graph_policy::sorted_merge : dyng::line_graph_policy::none;
  props.store_v2h = line_graph;
  return hgraph::from_hyperedges(res, list.view(), props);
}

tc::options kind_of(tc::pattern p) {
  tc::options o;
  o.kind = p;
  return o;
}

/// The oracle's count of one pattern on rows given by id.
[[maybe_unused]] std::uint64_t oracle(const rows& rs, std::int32_t n, tr::triad_pattern p) {
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int32_t> members;
  for (const auto& r : rs) {
    members.insert(members.end(), r.begin(), r.end());
    offsets.push_back(static_cast<std::int64_t>(members.size()));
  }
  return tr::triad_reference(offsets, members, n, p, {1}).at(0);
}

/// The three hyperedges of one type-3 triad whose pairwise-only regions have x members each.
rows type3_triad(std::int32_t x) {
  const row ab = range(0, x);
  const row bc = range(x, 2 * x);
  const row ca = range(2 * x, 3 * x);
  return {join(ab, ca), join(ab, bc), join(bc, ca)};
}

constexpr std::uint64_t choose3_2100000 = 1543497795000700000ULL;  // C(2,100,000, 3)
constexpr std::uint64_t choose3_3000000 = 4499995500001000000ULL;  // C(3,000,000, 3)
constexpr std::uint64_t choose3_3900000 = 9886492395001300000ULL;  // in (2^63, 2^64)
constexpr std::uint64_t type2_70000 = 12004828500000000000ULL;     // C(7e4, 2) * 7e4 * 7e4
constexpr std::uint64_t type3_2200000 = 10648000000000000000ULL;   // 2.2e6 ^ 3, in (2^63, 2^64)
constexpr std::uint64_t two_to_63 = std::uint64_t{1} << 63U;

class TriadCountLimits : public ::testing::TestWithParam<backend> {
 protected:
  [[nodiscard]] resources res() const {
    return dyng::test::make_resources(GetParam(), 4);
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

INSTANTIATE_TEST_SUITE_P(Backends, TriadCountLimits,
                         ::testing::ValuesIn(dyng::test::suite_backends()), backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TriadCountLimits);

TEST_P(TriadCountLimits, Type1OfHyperedgesOfMillionsOfMembersIsExact) {
  // C(2,100,000, 3) fits in 64 bits; n (n - 1) (n - 2) does not fit in 63.
  hgraph h = make(res(), 3'000'000, {range(0, 2'100'000)}, false);
  tc::result r = tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1));
  EXPECT_EQ(r.count(0), choose3_2100000);
  // Insert a hyperedge of 3,000,000 members, then delete the first.
  batch_t insert;
  const row three = range(0, 3'000'000);
  insert.insert_hyperedge(dyng::host_view(three));
  const tc::stats added = tc::update(res(), h, insert.view(), r);
  EXPECT_EQ(r.count(0), choose3_2100000 + choose3_3000000);
  EXPECT_EQ(added.triads_added, choose3_3000000);
  EXPECT_EQ(added.triads_removed, 0U);
  EXPECT_EQ(added.delta[0], static_cast<std::int64_t>(choose3_3000000));
  batch_t erase;
  erase.delete_hyperedge(0);
  const tc::stats removed = tc::update(res(), h, erase.view(), r);
  EXPECT_EQ(r.count(0), choose3_3000000);
  EXPECT_EQ(removed.triads_removed, choose3_2100000);
  EXPECT_EQ(removed.delta[0], -static_cast<std::int64_t>(choose3_2100000));
  EXPECT_EQ(r.count(0), tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1)).count(0));
}

TEST_P(TriadCountLimits, Type1PastTwoToThe64ThrowsCapacityError) {
  // One C(|e|, 3) past 2^64 - 1 (from 4,801,281 members).
  {
    const hgraph h = make(res(), 4'801'281, {range(0, 4'801'281)}, false);
    EXPECT_THROW((void)tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1)),
                 dyng::capacity_error);
  }
  // Two in (2^63, 2^64): each fits, the sum does not (the per-thread and the global sums).
  const row big = range(0, 3'900'000);
  {
    const hgraph h = make(res(), 3'900'000, {big, big}, false);
    EXPECT_THROW((void)tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1)),
                 dyng::capacity_error);
  }
  hgraph h = make(res(), 3'900'000, {big}, false);
  tc::result r = tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1));
  EXPECT_EQ(r.count(0), choose3_3900000);
  EXPECT_GT(r.count(0), two_to_63);
  // An update that takes the count past 2^64 - 1: capacity_error after the commit, r poisoned.
  batch_t b;
  b.insert_hyperedge(dyng::host_view(big));
  EXPECT_THROW((void)tc::update(res(), h, b.view(), r), dyng::capacity_error);
  EXPECT_THROW((void)r.counts(), dyng::stale_result_error);
}

TEST_P(TriadCountLimits, Type2BetweenTwoToThe63And64IsExactAndPastItRefused) {
  // e0 = [0, 2k), e1 = [0, k) + [2k, 3k): a = b = c = k.
  const auto pair = [](std::int32_t k) {
    return rows{range(0, 2 * k), join(range(0, k), range(2 * k, 3 * k))};
  };
  {
    const hgraph h = make(res(), 300'000, pair(100'000), true);
    EXPECT_THROW((void)tc::compute(res(), h, kind_of(tc::pattern::stathyper_type2)),
                 dyng::capacity_error);
  }
  const rows two = pair(70'000);
  hgraph h = make(res(), 210'000, two, true);
  tc::result r = tc::compute(res(), h, kind_of(tc::pattern::stathyper_type2));
  EXPECT_EQ(r.count(0), type2_70000);
  // A batch that removes and adds 1.2e19 each (delete e1, insert its copy): the phases are exact
  // magnitudes, the delta 0.
  batch_t swap;
  swap.delete_hyperedge(1);
  swap.insert_hyperedge(dyng::host_view(two[1]));
  const tc::stats s = tc::update(res(), h, swap.view(), r);
  EXPECT_EQ(s.triads_removed, type2_70000);
  EXPECT_EQ(s.triads_added, type2_70000);
  EXPECT_EQ(s.delta[0], 0);
  EXPECT_EQ(r.count(0), type2_70000);
  // Removing 1.2e19 on balance: the change is outside stats::delta's range on every backend.
  batch_t erase;
  erase.delete_hyperedge(s.batch.inserted_ids.at(0));
  EXPECT_THROW((void)tc::update(res(), h, erase.view(), r), dyng::capacity_error);
  EXPECT_THROW((void)r.counts(), dyng::stale_result_error);
  // Adding it on balance, likewise.
  hgraph one = make(res(), 210'000, {two[0]}, true);
  tc::result q = tc::compute(res(), one, kind_of(tc::pattern::stathyper_type2));
  EXPECT_EQ(q.count(0), 0U);
  batch_t add;
  add.insert_hyperedge(dyng::host_view(two[1]));
  EXPECT_THROW((void)tc::update(res(), one, add.view(), q), dyng::capacity_error);
  EXPECT_THROW((void)q.counts(), dyng::stale_result_error);
}

TEST_P(TriadCountLimits, Type3BetweenTwoToThe63And64IsExactAndPastItRefused) {
  {
    const hgraph h = make(res(), 6'600'000, type3_triad(2'200'000), true);
    EXPECT_EQ(tc::compute(res(), h, kind_of(tc::pattern::stathyper_type3)).count(0), type3_2200000);
  }
  const hgraph h = make(res(), 8'100'000, type3_triad(2'700'000), true);
  EXPECT_THROW((void)tc::compute(res(), h, kind_of(tc::pattern::stathyper_type3)),
               dyng::capacity_error);
}

#if !(defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA)

TEST(TriadReferenceLimits, ExactUpToTwoToThe64AndRefusedPastIt) {
  using P = tr::triad_pattern;
  EXPECT_EQ(oracle({range(0, 2'100'000)}, 2'100'000, P::stathyper_type1), choose3_2100000);
  EXPECT_EQ(oracle({range(0, 3'000'000)}, 3'000'000, P::stathyper_type1), choose3_3000000);
  EXPECT_EQ(oracle({range(0, 3'900'000)}, 3'900'000, P::stathyper_type1), choose3_3900000);
  EXPECT_EQ(oracle({range(0, 4'801'280)}, 4'801'280, P::stathyper_type1),
            18446738006366306560ULL);  // the largest C(n, 3) below 2^64
  EXPECT_THROW((void)oracle({range(0, 4'801'281)}, 4'801'281, P::stathyper_type1),
               std::overflow_error);
  EXPECT_THROW(
      (void)oracle({range(0, 3'900'000), range(0, 3'900'000)}, 3'900'000, P::stathyper_type1),
      std::overflow_error);
  EXPECT_EQ(oracle({range(0, 140'000), join(range(0, 70'000), range(140'000, 210'000))}, 210'000,
                   P::stathyper_type2),
            type2_70000);
  EXPECT_THROW((void)oracle({range(0, 200'000), join(range(0, 100'000), range(200'000, 300'000))},
                            300'000, P::stathyper_type2),
               std::overflow_error);
  EXPECT_EQ(oracle(type3_triad(2'200'000), 6'600'000, P::stathyper_type3), type3_2200000);
  EXPECT_THROW((void)oracle(type3_triad(2'700'000), 8'100'000, P::stathyper_type3),
               std::overflow_error);
}

#endif

}  // namespace
