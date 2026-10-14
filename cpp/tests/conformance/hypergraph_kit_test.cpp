// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_kit_test.cpp
 * @brief The conformance kit's hypergraph axis (ADR 0043; M8b criterion 11): the kit C0-C12 over
 *        the test-only aggregate-delta fake test_hyper_pairs (framework/fake_hyper_problems.hpp)
 *        on every backend of the executable, and the kit's own generator checks (a kit tests only
 *        what its generator makes: every hyperedge mix makes what its name promises). Compiled
 *        into the host executable (sequential, openmp) and, with DYNG_TEST_CUDA=1, into the CUDA
 *        one (cuda, compared with the host backends).
 */
#include "conformance/hypergraph_kit_traits.hpp"
// The traits before the kit (the suite reads them).
#include "conformance/conformance.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

DYNG_CONFORMANCE_SUITE(test_hyper_pairs);

namespace {

namespace dc = ::dyng::conformance;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using ct = dc::container_traits<hgraph>;

/// One batch of `mix` from a fresh model of `size`, applied to a sequential hypergraph under
/// `reuse` and escher_gpu().
struct applied_mix {
  ct::model_type before;          ///< the model before the batch
  ct::generated_type gen;         ///< the batch
  dyng::hyper_apply_summary sum;  ///< what the commit did
};

applied_mix apply_mix(dc::hyper_mix mix, dc::size_class size, dyng::id_reuse reuse) {
  std::mt19937_64 rng(77);
  applied_mix out;
  ct::model_type model = ct::random_model<dc::test_traits<dc::tags::test_hyper_pairs>>(size, rng);
  out.before = model;
  dyng::hypergraph_properties props;
  props.reuse = reuse;
  props.capacity = dyng::capacity_policy::escher_gpu();
  const dyng::resources res = dyng::resources::sequential();
  hgraph h = ct::build(model, res, props);
  out.gen = ct::random_batch(model, mix, rng, props);
  out.sum = h.apply(res, out.gen.batch.view());
  ct::observe(model, out.gen, out.sum);
  EXPECT_TRUE(ct::matches(h, model));
  return out;
}

TEST(HypergraphKitGenerators, EveryMixMakesWhatItsNamePromises) {
  using dc::hyper_mix;
  const auto erase_first = dyng::id_reuse::erase_first;
  {
    const applied_mix a = apply_mix(hyper_mix::insert_only, dc::size_class::small, erase_first);
    EXPECT_GT(a.sum.inserted_hyperedges, 0);
    EXPECT_EQ(a.sum.deleted_hyperedges, 0);
  }
  {
    const applied_mix a = apply_mix(hyper_mix::delete_only, dc::size_class::small, erase_first);
    EXPECT_GT(a.sum.deleted_hyperedges, 0);
    EXPECT_EQ(a.sum.inserted_hyperedges, 0);
  }
  {
    // Delete-then-reuse within one batch: under erase_first the new rows take the freed ids.
    const applied_mix a = apply_mix(hyper_mix::reuse, dc::size_class::small, erase_first);
    std::vector<std::int64_t> reused;
    for (const std::int64_t id : a.sum.inserted_ids) {
      if (std::find(a.gen.deleted.begin(), a.gen.deleted.end(), id) != a.gen.deleted.end()) {
        reused.push_back(id);
      }
    }
    EXPECT_FALSE(reused.empty()) << "no id was reused within the batch";
    // insert_first: the same batch cannot reuse them (the insert runs before the erase).
    const applied_mix b =
        apply_mix(hyper_mix::reuse, dc::size_class::small, dyng::id_reuse::insert_first);
    EXPECT_NE(a.sum.inserted_ids, b.sum.inserted_ids);
  }
  {
    const applied_mix a = apply_mix(hyper_mix::duplicates, dc::size_class::small, erase_first);
    EXPECT_GT(a.sum.dropped_members, 0);
    EXPECT_EQ(a.sum.dropped_members, a.gen.expected_dropped);
  }
  {
    const applied_mix a = apply_mix(hyper_mix::dead_ids, dc::size_class::small, erase_first);
    EXPECT_GE(a.sum.ignored_deletions, 4) << "repeated, dead, out-of-range and negative ids";
    EXPECT_EQ(a.sum.ignored_deletions, a.gen.expected_ignored);
  }
  {
    const applied_mix a = apply_mix(hyper_mix::empty, dc::size_class::small, erase_first);
    EXPECT_TRUE(a.gen.batch.view().empty());
  }
  {
    const applied_mix a = apply_mix(hyper_mix::delete_all, dc::size_class::small, erase_first);
    EXPECT_EQ(a.sum.deleted_hyperedges, static_cast<std::int64_t>(a.before.live.size()));
    EXPECT_EQ(a.sum.num_hyperedges_after, 0);
  }
  {
    const applied_mix a = apply_mix(hyper_mix::reinsert, dc::size_class::small, erase_first);
    ASSERT_EQ(a.gen.inserted.size(), a.gen.deleted_rows.size());
    for (std::size_t i = 0; i < a.gen.inserted.size(); ++i) {
      EXPECT_EQ(a.gen.inserted[i].members, a.gen.deleted_rows[i].members);
    }
  }
  {
    const applied_mix a = apply_mix(hyper_mix::larger, dc::size_class::small, erase_first);
    EXPECT_GT(a.sum.inserted_hyperedges, static_cast<std::int64_t>(a.before.live.size()))
        << "a batch larger than the hypergraph";
  }
  {
    const applied_mix a = apply_mix(hyper_mix::limits, dc::size_class::small, erase_first);
    EXPECT_GE(a.sum.ignored_deletions, 3) << "INT64_MIN, INT64_MAX and the bound";
    const std::int64_t n = a.before.num_vertices;
    bool low = false;
    bool high = false;
    for (const auto& r : a.gen.inserted) {
      low = low || std::find(r.members.begin(), r.members.end(), 0) != r.members.end();
      high = high || std::find(r.members.begin(), r.members.end(), n - 1) != r.members.end();
    }
    EXPECT_TRUE(low && high) << "members 0 and n - 1";
    EXPECT_EQ(a.gen.deleted.front(), a.before.live.rbegin()->first) << "the largest live id";
  }
  {
    // Cardinality >= 1,024 on the medium shape.
    const applied_mix a = apply_mix(hyper_mix::skewed, dc::size_class::medium, erase_first);
    std::size_t largest = 0;
    for (const auto& r : a.gen.inserted) largest = std::max(largest, r.members.size());
    EXPECT_GE(largest, 1024U);
  }
}

TEST(HypergraphKitGenerators, TheInverseRestoresTheMemberSets) {
  std::mt19937_64 rng(5);
  ct::model_type model =
      ct::random_model<dc::test_traits<dc::tags::test_hyper_pairs>>(dc::size_class::small, rng);
  const dyng::resources res = dyng::resources::sequential();
  hgraph h = ct::build(model, res, {});
  const auto sets = [&]() {
    const auto list = h.to_hyperedge_list(res);
    std::vector<std::vector<std::int32_t>> out;
    for (std::int64_t i = 0; i < list.num_hyperedges(); ++i) {
      out.emplace_back(list.members.begin() + list.offsets[static_cast<std::size_t>(i)],
                       list.members.begin() + list.offsets[static_cast<std::size_t>(i) + 1]);
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  const auto before = sets();
  const ct::generated_type gen = ct::random_batch(model, dc::hyper_mix::mixed, rng);
  const auto forward = h.apply(res, gen.batch.view());
  ct::observe(model, gen, forward);
  const ct::generated_type inverse = ct::inverse(model, gen, forward);
  ct::observe(model, inverse, h.apply(res, inverse.batch.view()));
  EXPECT_EQ(sets(), before);
  EXPECT_TRUE(ct::matches(h, model));
}

}  // namespace
