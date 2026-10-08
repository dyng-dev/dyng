// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:tests/unit/test_cbst_ops.cu
// Derived from MOSP_ESCHER@4b86159:tests/unit/test_cbst_smoke.cu
/**
 * @file cbst_scenarios_test.cpp
 * @brief MOSP_ESCHER's 9 CBST scenarios (test_cbst_ops) and its smoke test (test_cbst_smoke) on
 *        the dynG device store under each of its presets (M8a step 5, acceptance criterion 13).
 *
 * Every operation runs in lockstep on the store and on dyng::testing::cbst_host_model and the
 * two states must agree in every field (cbst_lockstep.hpp); the scenarios' own checks read the
 * store back with its integrity checker against the expected rows, as MOSP_ESCHER's harness did
 * after every step. Where a scenario depends on the policy (the churn scenarios), the store must
 * do what the preset says: escher_gpu() grows the payload and never compacts; mosp_escher()
 * compacts and throws only when the live rows really do not fit; dyng_default() compacts first
 * and grows when they do not.
 */
#include "parity/escher/cbst_lockstep.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

using row = lockstep_tree::row;
using rows = lockstep_tree::rows;

row iota_row(int from, int len) {
  row r(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) r[static_cast<std::size_t>(i)] = from + i;
  return r;
}

#define DYNG_SKIP_WITHOUT_DEVICE()                             \
  do {                                                         \
    if (!device_available()) GTEST_SKIP() << "no CUDA device"; \
  } while (false)

class CbstScenario : public ::testing::TestWithParam<const char*> {
 protected:
  std::string preset() const {
    return GetParam();
  }
};

#define EXPECT_CLEAN(h) EXPECT_TRUE((h).problems().empty()) << describe_problems((h).problems())

// scale: construct n = 1..1100 and n = 65,535 .. 2^20; every key must be found (the int32
// overflow of the in-order rank formula).
TEST_P(CbstScenario, Scale) {
  DYNG_SKIP_WITHOUT_DEVICE();
  // Reduced under the sanitizers (reduced_sizes()): n = 1..64 and 65,537.
  std::vector<int> sizes;
  for (int n = 1; n <= (reduced_sizes() ? 64 : 1100); ++n) sizes.push_back(n);
  if (reduced_sizes()) {
    sizes.push_back(65537);
  } else {
    for (int n : {65535, 65536, 65537, 70000, 131071, 200000, 1 << 20}) sizes.push_back(n);
  }
  for (int n : sizes) {
    rows values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
      values[static_cast<std::size_t>(i)] = iota_row(3 * i + 1, 1 + i % 3);
    lockstep_tree h(preset(), "scale", 12LL * n + 1024);
    h.construct(values);
    h.check("construct n=" + std::to_string(n));
    ASSERT_TRUE(h.problems().empty()) << describe_problems(h.problems());
  }
}

// reuse: erase, then reuse-insert twice; subtree_avail stays consistent and the keys are
// distinct and nonzero.
TEST_P(CbstScenario, Reuse) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const int n = 1000;
  std::mt19937_64 rng(7);
  rows values(n);
  for (int i = 0; i < n; ++i) values[static_cast<std::size_t>(i)] = iota_row(10 * i + 1, 1 + i % 5);
  lockstep_tree h(preset(), "reuse", 64LL * n);
  h.construct(values);
  std::set<int> victims;
  while (victims.size() < 59) victims.insert(1 + static_cast<int>(rng() % n));
  h.erase(std::vector<int>(victims.begin(), victims.end()));
  h.check("erase 59");
  rows items;
  for (int i = 0; i < 20; ++i) items.push_back(iota_row(100000 + 7 * i, 1));
  h.insert(items);
  h.check("reuse-insert 20");
  items.clear();
  for (int i = 0; i < 40; ++i) items.push_back(iota_row(200000 + 7 * i, 2));
  h.insert(items);
  h.check("insert 40 (39 reusable slots, 1 surplus)");
  EXPECT_CLEAN(h);
}

// terminator: rows whose length is a multiple of 4.
TEST_P(CbstScenario, Terminator) {
  DYNG_SKIP_WITHOUT_DEVICE();
  lockstep_tree h(preset(), "terminator", 4096);
  h.construct({{1, 2, 3, 4}, {5, 6}, iota_row(7, 8), {15, 16, 17}});
  h.check("construct");
  h.append({{1, {9}}, {3, {100, 101}}});
  h.check("fill rows of length 4 and 8");
  h.remove_values({{1, {2}}, {3, {8, 9}}});
  h.check("unfill");
  EXPECT_CLEAN(h);
}

// erase: erase an inner node, then fill, unfill and erase keys of its left subtree.
TEST_P(CbstScenario, Erase) {
  DYNG_SKIP_WITHOUT_DEVICE();
  rows values(15);
  for (int k = 1; k <= 15; ++k) values[static_cast<std::size_t>(k - 1)] = {10 * k};
  lockstep_tree h(preset(), "erase", 4096);
  h.construct(values);
  h.erase({8});  // the root of a 15-node CBST
  h.check("erase root key 8");
  h.append({{3, {999}}, {12, {777}}});
  h.check("fill keys 3 and 12");
  h.remove_values({{5, {50}}});
  h.check("unfill key 5");
  h.erase({3});
  h.check("erase key 3");
  EXPECT_CLEAN(h);
}

// surplus: a surplus insert rebuilds the tree; later fills must append.
TEST_P(CbstScenario, Surplus) {
  DYNG_SKIP_WITHOUT_DEVICE();
  lockstep_tree h(preset(), "surplus", 4096);
  h.construct({{10}, {20, 21, 22}, {30}});
  h.insert({{40, 41}});  // no deleted slot: surplus append + rebuild
  h.check("surplus insert");
  h.append({{2, {777}}, {1, {11, 12, 13, 14, 15, 16}}});
  h.check("fill after rebuild");
  h.append({{4, {42}}});
  h.check("fill the new row");
  EXPECT_CLEAN(h);
}

// bestfit: reused slots follow the best-fit matching and hold the whole item.
TEST_P(CbstScenario, BestFit) {
  DYNG_SKIP_WITHOUT_DEVICE();
  lockstep_tree h(preset(), "bestfit", 4096);
  h.construct({{10}, {20}, {30}, {40}, iota_row(500, 13), {60}});
  h.erase({2, 5});  // capacities 3 and 15
  h.check("erase keys 2 and 5");
  h.insert({iota_row(100, 10), {7, 8}});
  h.check("insert a 10-value and a 2-value item");
  EXPECT_CLEAN(h);
}

// unfill-chain: unfill on a chained row, then fill.
TEST_P(CbstScenario, UnfillChain) {
  DYNG_SKIP_WITHOUT_DEVICE();
  lockstep_tree h(preset(), "unfill-chain", 4096);
  h.construct({{1, 2}, {50}});
  h.append({{1, {3, 4, 5}}});  // overflows into a chained segment
  h.check("fill into a chain");
  h.remove_values({{1, {1}}});  // removal from the head segment
  h.check("unfill from the head segment");
  h.append({{1, {6}}});
  h.check("fill after unfill");
  h.remove_values({{1, {5, 6}}});
  h.append({{1, {7, 8, 9, 10, 11, 12, 13}}});
  h.check("unfill tail, fill again");
  EXPECT_CLEAN(h);
}

// random: random erase / unfill / fill / insert sequences with rows of 1..3,000 values (the
// thread, warp and block kernels), on both sides of 65,536 records.
TEST_P(CbstScenario, Random) {
  DYNG_SKIP_WITHOUT_DEVICE();
  std::array<std::int64_t, 9> bins{};
  // Reduced under the sanitizers: n = 300 and 5,000 (the block-level rows are kept).
  const std::vector<int> ns =
      reduced_sizes() ? std::vector<int>{300, 5000} : std::vector<int>{300, 5000, 70000};
  for (int n : ns) {
    SCOPED_TRACE("n=" + std::to_string(n));
    std::mt19937_64 rng(static_cast<std::uint64_t>(1234 + n));
    auto rnd = [&](int lo, int hi) { return lo + static_cast<int>(rng() % (hi - lo + 1)); };
    // Row lengths: mostly short, some warp-sized, a few block-sized.
    auto row_len = [&]() {
      const int r = rnd(0, 99);
      return r < 85 ? rnd(1, 12) : (r < 98 ? rnd(32, 200) : rnd(1024, 3000));
    };
    int next_value = 1;
    auto fresh_row = [&](int len) {
      row r(static_cast<std::size_t>(len));
      for (int& v : r) v = next_value++;
      return r;
    };
    rows values(static_cast<std::size_t>(n));
    for (auto& r : values) r = fresh_row(row_len());
    lockstep_tree h(preset(), "random", 400LL * n + (8 << 20));
    h.construct(values);
    h.check("construct");
    for (int round = 0; round < 6 && h.problems().empty(); ++round) {
      const std::string at = "round " + std::to_string(round);
      std::set<int> er;
      for (int i = 0; i < n / 25; ++i) {
        const int k = rnd(1, h.num_keys());
        if (h.live(k)) er.insert(k);
      }
      h.erase(std::vector<int>(er.begin(), er.end()), at + " erase");
      std::map<int, row> rem;
      for (int i = 0; i < n / 20; ++i) {
        const int k = rnd(1, h.num_keys());
        if (!h.live(k) || rem.count(k)) continue;
        const row& r = h.row_of(k);
        row vals;
        for (int v : r)
          if (rng() % 3 == 0) vals.push_back(v);
        if (vals.size() == r.size()) vals.pop_back();  // keep >= 1
        if (!vals.empty()) rem[k] = vals;
      }
      h.remove_values(rem, at + " unfill");
      std::map<int, row> add;
      for (int i = 0; i < n / 16; ++i) {
        const int k = rnd(1, h.num_keys());
        if (!h.live(k) || rem.count(k) || add.count(k)) continue;
        add[k] = fresh_row(row_len() / (rng() % 4 == 0 ? 1 : 3) + 1);
      }
      h.append(add, at + " fill");
      rows items;
      for (int i = 0; i < n / 20; ++i) items.push_back(fresh_row(row_len()));
      h.insert(items, at + " insert");
      h.check(at);
    }
    EXPECT_CLEAN(h);
    for (std::size_t i = 0; i < 9; ++i) bins[i] += h.bins()[i];
  }
  // Every kernel level of fill, reuse and unfill ran (the full sizes only: the block-level
  // unfill needs the rows of n = 70,000; dyng_cbst_coverage_tests covers every level too).
  if (reduced_sizes()) return;
  static const char* const names[9] = {"fill/thread",   "fill/warp",   "fill/block",
                                       "reuse/thread",  "reuse/warp",  "reuse/block",
                                       "unfill/thread", "unfill/warp", "unfill/block"};
  for (std::size_t i = 0; i < 9; ++i) EXPECT_GT(bins[i], 0) << names[i];
}

// churn 1: four rows of 8 values in 2,000 slots; every round each row gets 4 new values and
// loses its 4 oldest (MOSP_ESCHER's reported reproduction).
TEST_P(CbstScenario, ChurnSmall) {
  DYNG_SKIP_WITHOUT_DEVICE();
  rows values(4);
  int next = 1;
  for (auto& r : values) r = iota_row((next += 8) - 8, 8);
  lockstep_tree h(preset(), "churn-small", 2000);
  h.construct(values);
  for (int round = 0; round < 500 && h.problems().empty(); ++round) {
    std::map<int, row> add, rem;
    for (int k = 1; k <= 4; ++k) {
      add[k] = iota_row(next, 4);
      next += 4;
      rem[k].assign(h.row_of(k).begin(), h.row_of(k).begin() + 4);
    }
    h.append(add);
    h.remove_values(rem);
    if (round % 25 == 0 || round == 499) h.check("round " + std::to_string(round));
  }
  EXPECT_CLEAN(h);
  const auto [growths, compactions] = h.events();
  if (preset() == "escher_gpu") {
    EXPECT_EQ(compactions, 0);  // ESCHER-GPU grows its payload instead
    EXPECT_GT(growths, 0);
  } else {
    EXPECT_GT(compactions, 0);  // the space unfill frees is reclaimed
    EXPECT_EQ(growths, 0);      // the live rows always fit
    EXPECT_EQ(h.scalars().fixed_size, 2000);
  }
}

// churn 2: random balanced churn on 2,000 rows in 1.5x the initial payload: fills, unfills,
// erases and inserts (reused slots and surplus rows).
TEST_P(CbstScenario, ChurnRandom) {
  DYNG_SKIP_WITHOUT_DEVICE();
  std::mt19937_64 rng(99);
  auto rnd = [&](int lo, int hi) { return lo + static_cast<int>(rng() % (hi - lo + 1)); };
  int next_value = 1;
  auto fresh_row = [&](int len) {
    row r(static_cast<std::size_t>(len));
    for (int& v : r) v = next_value++;
    return r;
  };
  auto row_len = [&]() { return rnd(0, 9) < 9 ? rnd(1, 12) : rnd(32, 120); };
  const int n = 2000;
  rows values(n);
  long long flat = 0;
  for (auto& r : values) {
    r = fresh_row(row_len());
    flat += (static_cast<long long>(r.size()) + 4) / 4 * 4;
  }
  lockstep_tree h(preset(), "churn-random", flat * 3 / 2);
  h.construct(values);
  const int rounds = reduced_sizes() ? 40 : 300;  // reduced under the sanitizers
  for (int round = 0; round < rounds && h.problems().empty(); ++round) {
    std::map<int, row> add, rem;
    long long added = 0, removed = 0;
    for (int i = 0; i < n / 10; ++i) {
      const int k = rnd(1, h.num_keys());
      if (!h.live(k) || add.count(k)) continue;
      add[k] = fresh_row(rnd(1, 3) * (h.row_of(k).size() > 32 ? 8 : 1));
      added += static_cast<long long>(add[k].size());
    }
    for (int tries = 0; removed < added && tries < 4 * n; ++tries) {
      const int k = rnd(1, h.num_keys());
      if (!h.live(k) || add.count(k) || rem.count(k)) continue;
      const row& r = h.row_of(k);
      if (r.size() < 2) continue;
      const std::size_t take = std::min<std::size_t>(r.size() - 1, 1 + rng() % 8);
      rem[k].assign(r.begin(), r.begin() + static_cast<std::ptrdiff_t>(take));
      removed += static_cast<long long>(take);
    }
    h.append(add);
    h.remove_values(rem);
    std::set<int> er;
    rows items;
    for (int i = 0; i < 5; ++i) {
      const int k = rnd(1, h.num_keys());
      if (!h.live(k) || add.count(k) || rem.count(k) || !er.insert(k).second) continue;
      items.push_back(fresh_row(static_cast<int>(h.row_of(k).size())));
    }
    h.erase(std::vector<int>(er.begin(), er.end()));
    h.insert(items);
    if (round % 20 == 0 || round == rounds - 1) h.check("round " + std::to_string(round));
  }
  EXPECT_CLEAN(h);
  const auto [growths, compactions] = h.events();
  if (preset() == "escher_gpu") {
    EXPECT_EQ(compactions, 0);
    EXPECT_GT(growths, 0);
  } else {
    EXPECT_GT(compactions, 0);
  }
}

// churn 2b: surplus rows of an insert that do not fit behind the bump pointer.
TEST_P(CbstScenario, ChurnInsert) {
  DYNG_SKIP_WITHOUT_DEVICE();
  lockstep_tree h(preset(), "churn-insert", 200);
  h.construct({iota_row(1, 8), iota_row(100, 8)});
  int next = 1000;
  for (int round = 0; round < 100 && h.scalars().bump <= 200 - 40 && h.problems().empty();
       ++round) {
    std::map<int, row> add, rem;
    for (int k = 1; k <= 2; ++k) {
      add[k] = iota_row(next, 4);
      next += 4;
      rem[k].assign(h.row_of(k).begin(), h.row_of(k).begin() + 4);
    }
    h.append(add);
    h.remove_values(rem);
  }
  const long long before = h.scalars().bump;
  EXPECT_GT(before, 200 - 40) << "the bump pointer never came close to the capacity";
  const op_outcome out =
      h.insert({iota_row(5000, 8), iota_row(5100, 8), iota_row(5200, 8), iota_row(5300, 8)});
  EXPECT_EQ(out.status, op_status::ok) << out.message;
  h.check("insert 4 rows with the bump pointer at " + std::to_string(before) + " of 200");
  EXPECT_CLEAN(h);
  const auto [growths, compactions] = h.events();
  if (preset() == "escher_gpu") {
    EXPECT_EQ(compactions, 0);
  } else {
    EXPECT_GT(compactions, 0);
    EXPECT_EQ(growths, 0);
  }
}

// churn 3, growth: live data that no longer fits throws under MOSP_ESCHER's fixed capacity, and
// only then; escher_gpu() and dyng_default() grow the payload instead.
TEST_P(CbstScenario, ChurnGrowth) {
  DYNG_SKIP_WITHOUT_DEVICE();
  lockstep_tree h(preset(), "churn-growth", 256);
  h.construct({iota_row(1, 8), iota_row(100, 8)});
  int next = 1000;
  bool threw = false;
  long long live_at_throw = 0;
  std::string error;
  for (int round = 0; round < 200 && !threw && h.problems().empty(); ++round) {
    const op_outcome out = h.append({{1 + round % 2, iota_row(next, 3)}});
    if (out.status == op_status::error) {
      threw = true;
      error = out.error_type;
      live_at_throw = h.live_values();
    } else {
      next += 3;
    }
  }
  EXPECT_CLEAN(h);
  if (preset() == "mosp_escher") {
    // Rows of L values need at most L + 4 slots once compacted.
    ASSERT_TRUE(threw);
    EXPECT_EQ(error, "dyng::capacity_error");
    EXPECT_GT(live_at_throw + 3 + 2 * 4, 256);
  } else {
    EXPECT_FALSE(threw) << error;
    EXPECT_GT(h.scalars().fixed_size, 256);  // the payload grew
    h.check("after 200 rounds");
    EXPECT_CLEAN(h);
  }
}

// test_cbst_smoke: construct at sizes around the tree-position formula's corner cases, then
// erase the last record and insert a fresh one; both with the rows' counts and by the scan.
TEST_P(CbstScenario, Smoke) {
  DYNG_SKIP_WITHOUT_DEVICE();
  for (bool with_counts : {true, false}) {
    for (int n : {1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000}) {
      SCOPED_TRACE("n=" + std::to_string(n) + (with_counts ? " (counts)" : " (scan)"));
      rows values(static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i) values[static_cast<std::size_t>(i)] = {i + 1, (i + 1) * 10};
      lockstep_tree h(preset(), "smoke", 8192);
      h.construct(values, with_counts);
      h.erase({n});
      h.insert({{n + 1, (n + 1) * 10}});
      h.check("erase and insert");
      EXPECT_CLEAN(h);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Presets, CbstScenario,
                         ::testing::Values("escher_gpu", "mosp_escher", "dyng_default"),
                         [](const ::testing::TestParamInfo<const char*>& p) {
                           return std::string(p.param);
                         });

}  // namespace
}  // namespace dyng::test::cbst_compare
