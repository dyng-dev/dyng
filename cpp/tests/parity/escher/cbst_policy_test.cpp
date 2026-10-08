// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_policy_test.cpp
 * @brief The three presets of the dynG CBST store against the host model (M8a step 5, ADR 0039):
 *        random operation sequences under escher_gpu(), mosp_escher() and dyng_default() in
 *        lockstep with dyng::testing::cbst_host_model (whose dyng_default() is written
 *        independently of the store), the requested keys of an insert, and batches with many
 *        best-fit ties.
 *
 * Small payloads make every sequence grow (escher_gpu), compact (mosp_escher, dyng_default),
 * compact and then grow (dyng_default) or compact and then throw (mosp_escher); the store and the
 * model must agree on every outcome, InsertMapping and state field (cbst_lockstep.hpp).
 */
#include "parity/escher/cbst_lockstep.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

#define DYNG_SKIP_WITHOUT_DEVICE()                             \
  do {                                                         \
    if (!device_available()) GTEST_SKIP() << "no CUDA device"; \
  } while (false)

/// What the random sequences of one preset did.
struct sequence_totals {
  long long ops = 0, growths = 0, compactions = 0, errors = 0, inserts = 0;
};

/// One random sequence of `steps` operations on a small tree under `preset`.
void random_sequence(const std::string& preset, std::uint64_t seed, int steps,
                     sequence_totals& totals) {
  std::mt19937_64 rng(seed);
  auto rnd = [&](int lo, int hi) { return lo + static_cast<int>(rng() % (hi - lo + 1)); };
  int next_value = 1;
  auto fresh_row = [&](int len) {
    row r(static_cast<std::size_t>(len));
    for (int& v : r) v = next_value++;
    return r;
  };
  auto row_len = [&]() { return rnd(0, 19) == 0 ? rnd(32, 80) : rnd(1, 12); };
  const int n = rnd(0, 30);
  rows values(static_cast<std::size_t>(n));
  long long flat = 0;
  for (auto& r : values) {
    r = fresh_row(rnd(0, 9) == 0 ? 0 : row_len());
    flat += (static_cast<long long>(r.size()) + 4) / 4 * 4;
  }
  lockstep_tree h(preset, "random-" + std::to_string(seed), flat + flat / 2 + rnd(16, 160));
  h.construct(values, rnd(0, 1) == 1);
  bool failed = false;
  for (int step = 0; step < steps && h.problems().empty(); ++step) {
    const tree_digest d = h.device().digest(1);
    std::vector<int> live, dead;
    std::map<int, row> current;
    for (const key_digest& k : d.records) {
      (k.avail == 0 ? live : dead).push_back(k.key);
      current[k.key] = k.row;
    }
    const std::string at = "step " + std::to_string(step);
    const int pick = rnd(0, 99);
    op_outcome out;
    if (pick < 22 || live.empty()) {
      rows items;
      const int k = rnd(1, 5);
      for (int i = 0; i < k; ++i) items.push_back(fresh_row(rnd(0, 14) == 0 ? 0 : row_len()));
      out = h.insert(items, at + " insert");
      ++totals.inserts;
    } else if (pick < 52) {
      std::map<int, row> add;
      const int k = rnd(1, 6);
      for (int i = 0; i < k; ++i) add[live[rng() % live.size()]] = fresh_row(row_len());
      out = h.append(add, at + " fill");
    } else if (pick < 66) {
      std::set<int> er;
      const int k = rnd(1, 3);
      for (int i = 0; i < k; ++i) er.insert(live[rng() % live.size()]);
      if (!dead.empty() && rnd(0, 3) == 0) er.insert(dead[rng() % dead.size()]);  // again
      out = h.erase(std::vector<int>(er.begin(), er.end()), at + " erase");
    } else if (pick < 86) {
      std::map<int, row> rem;
      const int k = rnd(1, 4);
      for (int i = 0; i < k; ++i) {
        const int key = live[rng() % live.size()];
        row vals;
        for (int v : current[key])
          if (rng() % 3 == 0) vals.push_back(v);
        // A value the row does not hold, at a random place (not always last: a lost last value of
        // a removal list must show; MOSP_ESCHER's mutation #4).
        vals.insert(vals.begin() + static_cast<std::ptrdiff_t>(rng() % (vals.size() + 1)),
                    next_value + 1000000);
        rem[key] = vals;
      }
      out = h.remove_values(rem, at + " unfill");
    } else {
      out = h.compact(at + " compact");
    }
    ++totals.ops;
    if (out.status == op_status::error) {
      ++totals.errors;  // the store and the model agreed on it; the store is poisoned now
      failed = true;
      break;
    }
    if (step % 10 == 9) h.check(at);
  }
  if (!failed) h.check("end");  // a failed op left its in-place writes (and poisoned the store)
  const auto [g, c] = h.events();
  totals.growths += g;
  totals.compactions += c;
  EXPECT_TRUE(h.problems().empty()) << "seed " << seed << ": " << describe_problems(h.problems());
}

class CbstPresetLockstep : public ::testing::TestWithParam<const char*> {};

TEST_P(CbstPresetLockstep, RandomSequencesAgreeWithTheHostModel) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const std::string preset = GetParam();
  sequence_totals totals;
  for (std::uint64_t seed = 1; seed <= 40; ++seed) random_sequence(preset, seed, 60, totals);
  std::printf("%s: %lld ops, %lld inserts, %lld growths, %lld compactions, %lld errors\n",
              preset.c_str(), totals.ops, totals.inserts, totals.growths, totals.compactions,
              totals.errors);
  EXPECT_GT(totals.ops, preset == "mosp_escher" ? 500 : 2000);
  // The sequences reach what the preset does with a full payload.
  if (preset == "escher_gpu") {
    EXPECT_GT(totals.growths, 0);
    EXPECT_EQ(totals.errors, 0);
  } else if (preset == "mosp_escher") {
    EXPECT_EQ(totals.growths, 0);
    EXPECT_GT(totals.compactions, 0);
    EXPECT_GT(totals.errors, 0);  // live rows that did not fit
  } else {
    EXPECT_GT(totals.growths, 0);
    EXPECT_GT(totals.compactions, 0);
    EXPECT_EQ(totals.errors, 0);
  }
}

// The requested keys of an insert are ignored (both originals assign keys by best fit, then
// after the last key): permuted, garbage (0, negative, duplicate, huge) and absent requested keys
// give the same InsertMapping and the same state.
TEST_P(CbstPresetLockstep, RequestedKeysDoNotChangeTheMapping) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const std::string preset = GetParam();
  const rows items = {{1, 2, 3}, {4}, {5, 6, 7, 8, 9}, {10, 11}, {12}, {13, 14, 15, 16}};
  const std::vector<std::vector<int>> requested = {
      {31, 32, 33, 34, 35, 36},          // what MOSP_ESCHER's harness passes
      {36, 34, 35, 31, 33, 32},          // permuted
      {0, -1, 7, 7, INT32_MAX, -99999},  // garbage
      {1, 2, 3, 4, 5, 6}};               // live keys
  std::vector<int> first_mapping;
  tree_digest first_state;
  for (std::size_t r = 0; r < requested.size(); ++r) {
    SCOPED_TRACE(r);
    lockstep_tree h(preset, "requested", 4096);
    rows values;
    for (int i = 0; i < 30; ++i)
      values.push_back(row(static_cast<std::size_t>(1 + i % 6), 100 + i));
    h.construct(values);
    h.erase({3, 7, 11, 19, 23});
    const op_outcome out = h.insert(items, "insert", &requested[r]);
    ASSERT_EQ(out.status, op_status::ok) << out.message;
    h.check("after the insert");
    EXPECT_TRUE(h.problems().empty()) << describe_problems(h.problems());
    const tree_digest state = h.device().digest(1);
    if (r == 0) {
      first_mapping = out.mapping;
      first_state = state;
      continue;
    }
    EXPECT_EQ(out.mapping, first_mapping);
    EXPECT_EQ(digest_difference(state, first_state), "");
  }
}

// Batches with many ties: equal slot lengths and equal item sizes, reused and surplus, matched
// by the store as by the host model (the best-fit tie order of both originals).
TEST_P(CbstPresetLockstep, ManyTiesMatchTheHostModel) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const std::string preset = GetParam();
  for (int round = 0; round < 3; ++round) {
    SCOPED_TRACE(round);
    std::mt19937_64 rng(static_cast<std::uint64_t>(17 + round));
    lockstep_tree h(preset, "ties", 1 << 16);
    rows values;
    int next = 1;
    for (int i = 0; i < 400; ++i) {
      const int len = 1 + (i % 3) * 4;  // three row lengths: many equal slots
      row r(static_cast<std::size_t>(len));
      for (int& v : r) v = next++;
      values.push_back(r);
    }
    h.construct(values);
    std::set<int> er;
    while (er.size() < 150) er.insert(1 + static_cast<int>(rng() % 400));
    h.erase(std::vector<int>(er.begin(), er.end()));
    rows items;
    for (int i = 0; i < 220; ++i) {  // more items than slots: ties among items and surplus rows
      const int len = 1 + static_cast<int>(rng() % 3) * 4;
      row r(static_cast<std::size_t>(len));
      for (int& v : r) v = next++;
      items.push_back(r);
    }
    const op_outcome out = h.insert(items, "insert 220 tied items");
    ASSERT_EQ(out.status, op_status::ok) << out.message;
    h.check("after the insert");
    EXPECT_TRUE(h.problems().empty()) << describe_problems(h.problems());
  }
}

INSTANTIATE_TEST_SUITE_P(Presets, CbstPresetLockstep,
                         ::testing::Values("escher_gpu", "mosp_escher", "dyng_default"),
                         [](const ::testing::TestParamInfo<const char*>& p) {
                           return std::string(p.param);
                         });

}  // namespace
}  // namespace dyng::test::cbst_compare
