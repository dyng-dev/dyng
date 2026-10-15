// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_store_test.cpp
 * @brief The host best-fit id allocator against dyng::testing::cbst_host_model (two independent
 *        implementations of the H2V tree's ids, slots and payload accounting), and the mapping of
 *        capacity_policy to the store's presets (M8b, ADR 0041, sections 5 and 6).
 */
#include "hypergraph/capacity.hpp"
#include "hypergraph/escher/store_policy.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "hypergraph/id_allocator.hpp"
#include "hypergraph/store.hpp"

#include <dyng/core/error.hpp>
#include <dyng/testing/cbst_host_model.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::capacity_policy;
using dyng::detail::hyper::id_allocator;
namespace esc = dyng::detail::escher;

std::vector<std::string> presets() {
  return {"escher_gpu", "mosp_escher", "dyng_default"};
}

capacity_policy capacity_of(const std::string& preset, std::int64_t extra) {
  if (preset == "escher_gpu") return capacity_policy::escher_gpu();
  if (preset == "mosp_escher") return capacity_policy::mosp_escher(extra);
  return capacity_policy{};
}

// The allocator and the model after the same operation: every key's state and slot, the bump
// pointer, the capacity, the compactions.
std::string difference(const id_allocator& a, const dyng::testing::cbst_host_model& m) {
  std::string out;
  std::int64_t live = 0, dead = 0;
  for (const auto& rec : m.records()) {
    const auto state = a.state_of(rec.key);
    const auto want =
        rec.avail != 0 ? id_allocator::key_state::dead : id_allocator::key_state::live;
    if (state != want) out += "key " + std::to_string(rec.key) + " state; ";
    if (a.length(rec.key) != rec.length) out += "key " + std::to_string(rec.key) + " length; ";
    if (rec.avail == 0 && a.count(rec.key) != rec.row_len)
      out += "key " + std::to_string(rec.key) + " count; ";
    (rec.avail != 0 ? dead : live) += 1;
  }
  if (live != a.live() || dead != a.dead()) out += "key counts; ";
  if (a.bump() != m.bump()) {
    out += "bump " + std::to_string(a.bump()) + " vs " + std::to_string(m.bump()) + "; ";
  }
  if (a.capacity() != m.payload_capacity()) {
    out += "capacity " + std::to_string(a.capacity()) + " vs " +
           std::to_string(m.payload_capacity()) + "; ";
  }
  std::int64_t compactions = 0;
  for (const auto& e : m.events())
    compactions += e.what == dyng::testing::cbst_model_event::kind::compaction ? 1 : 0;
  if (compactions != a.compactions()) out += "compactions; ";
  for (const auto& p : a.check()) out += "allocator: " + p + "; ";
  return out;
}

// Random insert / erase / append streams on the allocator and the model, under every preset
// (small capacities, so that compaction and growth happen).
TEST(IdAllocator, MatchesTheHostModelUnderEveryPreset) {
  for (const std::string& preset : presets()) {
    std::int64_t compactions = 0, growths = 0, reuses = 0, rebuilds = 0;
    for (const std::uint64_t seed : {1ULL, 2ULL, 3ULL, 4ULL}) {
      SCOPED_TRACE(preset + " seed " + std::to_string(seed));
      std::mt19937_64 rng(seed);
      const capacity_policy c = capacity_of(preset, 64);
      // The initial rows: 40 rows of 1..9 values.
      std::vector<std::int32_t> counts;
      std::vector<std::int32_t> keys, starts, flat, row_counts;
      std::uniform_int_distribution<std::int32_t> size(1, 9);
      for (int i = 0; i < 40; ++i) counts.push_back(size(rng));
      for (std::size_t i = 0; i < counts.size(); ++i) {
        keys.push_back(static_cast<std::int32_t>(i) + 1);
        starts.push_back(static_cast<std::int32_t>(flat.size()));
        const std::int32_t padded = (counts[i] + 4) / 4 * 4;
        for (std::int32_t k = 0; k < counts[i]; ++k) flat.push_back(k + 1);
        flat.insert(flat.end(), static_cast<std::size_t>(padded - counts[i] - 1), 0);
        flat.push_back(INT_MIN);
      }
      const std::int64_t capacity =
          preset == "escher_gpu"
              ? static_cast<std::int64_t>(flat.size()) + 8
              : dyng::detail::hyper::initial_payload(c, static_cast<std::int64_t>(flat.size()));
      id_allocator a(dyng::detail::hyper::to_store_policy(c), capacity);
      a.construct(counts);
      dyng::testing::cbst_host_model m(dyng::test::hyper::model_policy(preset), capacity, 4, "h2v");
      row_counts = counts;
      ASSERT_EQ(m.construct(keys, starts, flat, &row_counts).status,
                dyng::testing::cbst_op_status::ok);
      ASSERT_EQ(difference(a, m), "");
      for (int op = 0; op < 400; ++op) {
        const int kind = static_cast<int>(rng() % 3);
        std::vector<std::int32_t> live;
        for (const auto& rec : m.records())
          if (rec.avail == 0) live.push_back(rec.key);
        if (kind == 0 && !live.empty()) {  // erase a few
          std::shuffle(live.begin(), live.end(), rng);
          live.resize(std::min<std::size_t>(live.size(), 1 + rng() % 6));
          a.erase(live);
          ASSERT_EQ(m.erase(live).status, dyng::testing::cbst_op_status::ok);
        } else if (kind == 1) {  // insert a few (sizes with ties)
          std::vector<std::int32_t> sizes, requested, payload, prefix;
          const int k = 1 + static_cast<int>(rng() % 7);
          for (int i = 0; i < k; ++i) {
            sizes.push_back(1 + static_cast<std::int32_t>(rng() % (op % 50 == 0 ? 40 : 8)));
            requested.push_back(i + 1);
            for (std::int32_t v = 0; v < sizes.back(); ++v) payload.push_back(v + 1);
            prefix.push_back(static_cast<std::int32_t>(payload.size()));
          }
          std::vector<std::int32_t> got;
          std::vector<std::int32_t> dead_before;
          for (std::int64_t key = 1; key <= a.max_key(); ++key)
            if (a.state_of(key) == id_allocator::key_state::dead)
              dead_before.push_back(static_cast<std::int32_t>(key));
          bool threw = false;
          try {
            got = a.insert(sizes);
          } catch (const dyng::capacity_error&) {
            threw = true;
          }
          const auto r = m.insert(requested, payload, prefix);
          ASSERT_EQ(threw, r.status != dyng::testing::cbst_op_status::ok) << r.message;
          if (threw) break;  // both are undefined after a failure (the container poisons)
          ASSERT_EQ(got, r.mapping) << "op " << op;
          std::int64_t reused = 0;
          for (const std::int32_t key : got)
            reused += std::binary_search(dead_before.begin(), dead_before.end(), key) ? 1 : 0;
          reuses += reused;
          // A surplus rebuild that dropped deleted slots it did not reuse.
          rebuilds += reused < static_cast<std::int64_t>(got.size()) &&
                      static_cast<std::int64_t>(dead_before.size()) > reused;
        } else if (!live.empty()) {  // append to a few live rows (padding rule 3)
          std::shuffle(live.begin(), live.end(), rng);
          live.resize(std::min<std::size_t>(live.size(), 1 + rng() % 4));
          std::sort(live.begin(), live.end());
          std::vector<std::int32_t> adds, payload, prefix;
          for (std::size_t i = 0; i < live.size(); ++i) {
            adds.push_back(1 + static_cast<std::int32_t>(rng() % 9));
            for (std::int32_t v = 0; v < adds.back(); ++v) payload.push_back(100 + v);
            prefix.push_back(static_cast<std::int32_t>(payload.size()));
          }
          bool threw = false;
          try {
            a.append(live, adds);
          } catch (const dyng::capacity_error&) {
            threw = true;
          }
          const auto r = m.append(live, payload, prefix);
          ASSERT_EQ(threw, r.status != dyng::testing::cbst_op_status::ok) << r.message;
          if (threw) break;
        }
        ASSERT_EQ(difference(a, m), "") << "op " << op << " kind " << kind;
      }
      compactions += a.compactions();
      growths += a.growths();
    }
    // The streams reach every rule they are meant to test.
    EXPECT_GT(reuses, 0) << preset;
    EXPECT_GT(rebuilds, 0) << preset;
    if (preset != "escher_gpu") {
      EXPECT_GT(compactions, 0) << preset;
    }
    if (preset != "mosp_escher") {
      EXPECT_GT(growths, 0) << preset;
    }
  }
}

TEST(IdAllocator, StateRoundTrip) {
  id_allocator a(esc::store_policy::escher_gpu(), 1000);
  a.construct({3, 1, 5, 2, 7});
  a.erase({2, 4});
  (void)a.insert({9});
  const auto s = a.state();
  id_allocator b(esc::store_policy::escher_gpu(), 0);
  b.restore(s);
  EXPECT_EQ(b.max_key(), a.max_key());
  EXPECT_EQ(b.live(), a.live());
  EXPECT_EQ(b.bump(), a.bump());
  EXPECT_EQ(b.insert({1, 1, 6}), a.insert({1, 1, 6}));
}

TEST(CapacityPolicy, PresetsMapToTheStorePresets) {
  const auto same = [](const esc::store_policy& x, const esc::store_policy& y) {
    EXPECT_EQ(x.name, y.name);
    EXPECT_EQ(x.growth, y.growth);
    EXPECT_EQ(x.compaction, y.compaction);
    EXPECT_EQ(x.scratch, y.scratch);
    EXPECT_EQ(x.alignment, y.alignment);
    EXPECT_EQ(x.max_payload, y.max_payload);
    EXPECT_EQ(x.compaction_headroom, y.compaction_headroom);
  };
  same(dyng::detail::hyper::to_store_policy(capacity_policy::escher_gpu()),
       esc::store_policy::escher_gpu());
  same(dyng::detail::hyper::to_store_policy(capacity_policy::escher_gpu(5000)),
       esc::store_policy::escher_gpu());
  same(dyng::detail::hyper::to_store_policy(capacity_policy::mosp_escher(12345)),
       esc::store_policy::mosp_escher(12345));
  same(dyng::detail::hyper::to_store_policy(capacity_policy{}), esc::store_policy::dyng_default());
}

TEST(CapacityPolicy, InitialSizingIsTheStoresAtBoundaries) {
  // Every size class: small, the multiples of 10 and 8 (exact products), and the sizes near the
  // caps (MOSP_ESCHER's 2.0e9, the int32 offsets).
  std::vector<std::int64_t> sizes;
  for (std::int64_t s = 0; s < 5000; ++s) sizes.push_back(s);
  for (std::int64_t s = 1; s < 2000; ++s) {
    sizes.push_back(s * 1000003);
    sizes.push_back(s * 1000000);
    sizes.push_back(1538461530 + s);  // around 2.0e9 / 1.3
  }
  for (std::int64_t s = 1900000000; s < 1900000000 + 20000; s += 10) sizes.push_back(s);
  const auto a = capacity_policy::escher_gpu(4096);
  const auto b = capacity_policy::mosp_escher(1 << 20);
  const auto sa = esc::store_policy::escher_gpu();
  const auto sb = esc::store_policy::mosp_escher(1 << 20);
  for (const std::int64_t s : sizes) {
    // ESCHER-GPU: max(requested, size + size / 8 + 1024) for in-place trees (below the cap).
    const std::int64_t want_a = std::max<std::int64_t>(4096, s + s / 8 + 1024);
    if (want_a <= INT_MAX) {
      ASSERT_EQ(dyng::detail::hyper::initial_payload(a, s), sa.initial_capacity(4096, s, true))
          << s;
    }
    // MOSP_ESCHER: floor(size * 1.3) + extra, at most 2.0e9.
    try {
      const std::int64_t got = dyng::detail::hyper::initial_payload(b, s);
      ASSERT_EQ(got, sb.initial_capacity(0, s, true)) << s;
      ASSERT_EQ(
          got, std::min<std::int64_t>(
                   static_cast<std::int64_t>(static_cast<double>(s) * 1.3) + (1 << 20), 2000000000))
          << s;
    } catch (const dyng::capacity_error&) {
      EXPECT_THROW((void)sb.initial_capacity(0, s, true), dyng::capacity_error) << s;
    }
  }
}

// MOSP_ESCHER's tests set other Caps factors (2.0, 1.25, 1.5): initial_headroom = factor - 1
// gives floor(size * factor) exactly.
TEST(CapacityPolicy, OtherMospEscherFactorsAreExact) {
  for (const double factor : {2.0, 1.25, 1.5}) {
    auto c = capacity_policy::mosp_escher(4096);
    c.initial_headroom = factor - 1.0;
    for (std::int64_t s = 0; s < 1500000000; s = s * 3 + 7) {
      const std::int64_t want = std::min<std::int64_t>(
          static_cast<std::int64_t>(static_cast<double>(s) * factor) + 4096, 2000000000);
      if (want < s) continue;
      ASSERT_EQ(dyng::detail::hyper::initial_payload(c, s), want) << factor << " " << s;
    }
  }
}

TEST(CapacityPolicy, Validation) {
  capacity_policy c;
  c.grow_factor = 2.0;
  EXPECT_THROW(dyng::detail::hyper::validate_capacity(c), dyng::not_supported_error);
  c = capacity_policy{};
  c.initial_headroom = -1;
  EXPECT_THROW(dyng::detail::hyper::validate_capacity(c), dyng::invalid_argument_error);
  c = capacity_policy{};
  c.row_alignment = 0;
  EXPECT_THROW(dyng::detail::hyper::validate_capacity(c), dyng::invalid_argument_error);
  c = capacity_policy{};
  c.max_payload = 100;
  EXPECT_THROW((void)dyng::detail::hyper::initial_payload(c, 101), dyng::capacity_error);
  EXPECT_EQ(dyng::detail::hyper::initial_payload(c, 50), 100);
  // M8b review: a huge slack or headroom saturates at the payload limit (no int64 overflow, no
  // out-of-range double conversion, no false capacity_error).
  c = capacity_policy{};
  c.initial_slack = INT64_MAX;
  EXPECT_EQ(dyng::detail::hyper::initial_payload(c, 8), INT_MAX);
  c = capacity_policy{};
  c.initial_headroom = 1e300;
  EXPECT_EQ(dyng::detail::hyper::initial_payload(c, 8), INT_MAX);
  c.max_payload = 1000;
  EXPECT_EQ(dyng::detail::hyper::initial_payload(c, 8), 1000);
  c = capacity_policy{};
  c.initial_slack = INT64_MAX;
  c.initial_headroom = 1e300;
  c.max_payload = 64;
  EXPECT_EQ(dyng::detail::hyper::initial_payload(c, 0), 64);
  EXPECT_THROW((void)dyng::detail::hyper::initial_payload(c, 65), dyng::capacity_error);
}

TEST(GroupByVertex, IsEscherGpusLayout) {
  dyng::detail::hyper::member_rows r;
  r.members = {3, 1, 2, 1, 3};
  r.offsets = {0, 2, 3, 5};
  dyng::detail::hyper::grouped_incidences g;
  dyng::detail::hyper::group_by_vertex(r, {10, 20, 30}, g);
  EXPECT_EQ(g.vertices, (std::vector<std::int32_t>{1, 2, 3}));
  EXPECT_EQ(g.ids, (std::vector<std::int64_t>{10, 30, 20, 10, 30}));
  EXPECT_EQ(g.prefix, (std::vector<std::int32_t>{2, 3, 5}));
}

}  // namespace
