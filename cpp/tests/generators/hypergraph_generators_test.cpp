// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_generators_test.cpp
 * @brief generators/hypergraphs.hpp: the streams are pinned by golden hashes (bit-exact per (seed,
 *        major version)), every mix does what its documentation says and applies to a hypergraph;
 *        generators::legacy's ESCHER-GPU generators and the run-time check of the standard
 *        library (their parity with the originals' dumps is in dyng_hypergraph_ids_tests).
 */
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/hypergraphs.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace gen = dyng::generators;
using hgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using ugraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;

std::uint64_t fnv(std::uint64_t h, std::int64_t v) {
  for (int b = 0; b < 8; ++b) {
    h ^= static_cast<std::uint64_t>(v) & 0xffU;
    h *= 0x100000001b3ULL;
    v = static_cast<std::int64_t>(static_cast<std::uint64_t>(v) >> 8U);
  }
  return h;
}

template <typename range_t>
std::uint64_t fnv_all(std::uint64_t h, const range_t& r) {
  h = fnv(h, static_cast<std::int64_t>(r.size()));
  for (const auto x : r) h = fnv(h, static_cast<std::int64_t>(x));
  return h;
}

template <typename weight_t>
std::uint64_t hash_of(const dyng::hyperedge_list<std::int32_t, weight_t>& l) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  h = fnv(h, l.num_vertices);
  h = fnv_all(h, l.offsets);
  h = fnv_all(h, l.members);
  if constexpr (!dyng::is_unweighted_v<weight_t>) h = fnv_all(h, l.weights);
  return h;
}

template <typename weight_t>
std::uint64_t hash_of(const dyng::hyperedge_batch<std::int32_t, weight_t>& b) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  h = fnv_all(h, b.delete_hyperedges());
  h = fnv_all(h, b.insert_offsets());
  h = fnv_all(h, b.insert_members());
  if constexpr (!dyng::is_unweighted_v<weight_t>) h = fnv_all(h, b.insert_weights());
  return h;
}

gen::uniform_hypergraph_options base(std::uint64_t seed) {
  gen::uniform_hypergraph_options o;
  o.num_vertices = 1500;
  o.num_hyperedges = 400;
  o.min_cardinality = 1;
  o.max_cardinality = 9;
  o.max_weight = 50;
  o.seed = seed;
  return o;
}

std::vector<gen::hyperedge_batch_mix> all_mixes() {
  using m = gen::hyperedge_batch_mix;
  return {m::insert_only, m::delete_only, m::mixed,  m::reuse,      m::heavy_delete,
          m::delete_all,  m::reinsert,    m::larger, m::duplicates, m::dead_ids,
          m::skewed,      m::limits,      m::empty};
}

// The golden hashes: the streams of 0.3 (bit-exact per (seed, major version), PLAN 5.8). Change
// them only with a major version (a stream change is a breaking change, PLAN 5.9).
TEST(UniformHypergraph, StreamsAreBitExactPerSeed) {
  const std::uint64_t golden[3] = {0x851e37c4f9a8ca6fULL, 0x59488d8ff3b9526fULL,
                                   0xe418d964c169bd23ULL};
  const std::uint64_t golden_unweighted = 0x71dc3220c3a72978ULL;
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    const auto l = gen::uniform_hypergraph<std::int32_t, std::int32_t>(base(seed));
    EXPECT_EQ(hash_of(l), golden[seed - 1]) << seed;
  }
  const auto u = gen::uniform_hypergraph<std::int32_t, dyng::unweighted>(base(7));
  EXPECT_EQ(hash_of(u), golden_unweighted);
}

TEST(UniformHypergraph, DrawsDistinctSortedMembersInTheRanges) {
  auto o = base(11);
  o.num_vertices = 12;
  o.num_hyperedges = 2000;
  o.min_cardinality = 2;
  o.max_cardinality = 20;  // clamped to n
  o.min_weight = 3;
  o.max_weight = 4;
  const auto l = gen::uniform_hypergraph<std::int32_t, std::int32_t>(o);
  ASSERT_EQ(l.num_hyperedges(), 2000);
  EXPECT_EQ(l.num_vertices, 12);
  std::set<std::int64_t> sizes;
  for (std::int64_t i = 0; i < l.num_hyperedges(); ++i) {
    const std::vector<std::int32_t> r(l.members.begin() + l.offsets[i],
                                      l.members.begin() + l.offsets[i + 1]);
    EXPECT_TRUE(std::is_sorted(r.begin(), r.end()));
    EXPECT_EQ(std::adjacent_find(r.begin(), r.end()), r.end());
    EXPECT_GE(r.front(), 0);
    EXPECT_LT(r.back(), 12);
    sizes.insert(static_cast<std::int64_t>(r.size()));
  }
  EXPECT_EQ(*sizes.begin(), 2);
  EXPECT_EQ(*sizes.rbegin(), 12);
  for (const auto w : l.weights) EXPECT_TRUE(w == 3 || w == 4);
  // The same options give the same stream.
  EXPECT_EQ(hash_of(l), hash_of(gen::uniform_hypergraph<std::int32_t, std::int32_t>(o)));
  // Invalid options.
  o.min_cardinality = 0;
  EXPECT_THROW((void)(gen::uniform_hypergraph<std::int32_t, std::int32_t>(o)),
               dyng::invalid_argument_error);
  o = base(1);
  o.num_vertices = 0;
  EXPECT_THROW((void)(gen::uniform_hypergraph<std::int32_t, std::int32_t>(o)),
               dyng::invalid_argument_error);
  o = base(1);
  o.max_weight = 0;
  EXPECT_THROW((void)(gen::uniform_hypergraph<std::int32_t, std::int32_t>(o)),
               dyng::invalid_argument_error);
  o = base(1);
  o.num_hyperedges = 0;
  o.num_vertices = 0;
  EXPECT_EQ((gen::uniform_hypergraph<std::int32_t, std::int32_t>(o).num_hyperedges()), 0);
}

class HyperedgeBatches : public ::testing::Test {
 protected:
  // A hypergraph with dead ids: 400 hyperedges, then every third one deleted.
  hgraph start() const {
    hgraph h = hgraph::from_hyperedges(
        res_, gen::uniform_hypergraph<std::int32_t, std::int32_t>(base(5)).view(), props_);
    dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
    for (std::int64_t id = 0; id < 400; id += 3) b.delete_hyperedge(id);
    (void)h.apply(res_, b.view());
    return h;
  }
  dyng::resources res_ = dyng::resources::sequential();
  dyng::hypergraph_properties props_ = [] {
    dyng::hypergraph_properties p;
    p.reuse = dyng::id_reuse::erase_first;
    p.capacity = dyng::capacity_policy::escher_gpu();
    return p;
  }();
};

TEST_F(HyperedgeBatches, StreamsAreBitExactPerSeed) {
  const std::vector<std::uint64_t> golden = {
      0x8c354b9f59848e81ULL,  // insert_only
      0x33abafe9519d3decULL,  // delete_only
      0x8b3f9cede10153acULL,  // mixed
      0x37a8346db2fed788ULL,  // reuse
      0x130e874382750e87ULL,  // heavy_delete
      0x51743417045408e3ULL,  // delete_all
      0x85d110edbca3d907ULL,  // reinsert
      0x8b691913e6da4175ULL,  // larger
      0xd824a03e219c0f28ULL,  // duplicates
      0x6c57023c6b81451eULL,  // dead_ids
      0x9fb499c68d3cb708ULL,  // skewed
      0x824b193ec2c94cfdULL,  // limits
      0x0c8210784d8af5a5ULL,  // empty
  };
  const hgraph h = start();
  const auto mixes = all_mixes();
  ASSERT_EQ(golden.size(), mixes.size());
  for (std::size_t i = 0; i < mixes.size(); ++i) {
    gen::hyperedge_batch_options o;
    o.mix = mixes[i];
    o.seed = 42;
    o.max_cardinality = 6;
    o.max_weight = 9;
    const auto b = gen::random_hyperedge_batch(res_, h, o);
    EXPECT_EQ(hash_of(b), golden[i]) << gen::to_string(mixes[i]);
  }
}

TEST_F(HyperedgeBatches, EveryMixDoesWhatItSaysAndApplies) {
  const hgraph h0 = start();
  const std::int64_t live = h0.num_hyperedges();
  const std::int64_t bound = h0.hyperedge_id_bound();
  ASSERT_EQ(live, 266);
  for (const auto mix : all_mixes()) {
    SCOPED_TRACE(std::string(gen::to_string(mix)));
    hgraph h = h0.clone(res_);
    gen::hyperedge_batch_options o;
    o.mix = mix;
    o.seed = 9;
    o.size = 20;
    o.skewed_cardinality = 1100;
    const auto b = gen::random_hyperedge_batch(res_, h, o);
    const auto& del = b.delete_hyperedges();
    const std::set<std::int64_t> distinct(del.begin(), del.end());
    using m = gen::hyperedge_batch_mix;
    switch (mix) {
      case m::insert_only:
        EXPECT_EQ(b.num_insertions(), 20u);
        EXPECT_EQ(del.size(), 0u);
        break;
      case m::delete_only:
        EXPECT_EQ(distinct.size(), 20u);
        EXPECT_EQ(b.num_insertions(), 0u);
        break;
      case m::mixed:
        EXPECT_EQ(del.size(), 10u);
        EXPECT_EQ(b.num_insertions(), 10u);
        break;
      case m::reuse:
        EXPECT_EQ(del.size(), 20u);
        EXPECT_EQ(b.num_insertions(), 20u);
        break;
      case m::heavy_delete:
        EXPECT_EQ(static_cast<std::int64_t>(distinct.size()), live - live / 10);
        break;
      case m::delete_all:
        EXPECT_EQ(static_cast<std::int64_t>(distinct.size()), live);
        break;
      case m::reinsert:
        EXPECT_EQ(b.num_insertions(), 20u);
        break;
      case m::larger:
        EXPECT_GT(static_cast<std::int64_t>(b.num_insertions()), live);
        break;
      case m::duplicates:
        EXPECT_EQ(b.num_insertions(), 21u);
        break;
      case m::dead_ids:
        EXPECT_EQ(del.size(), 7u);  // 2 live, the same 2, a dead id, the bound, -1
        EXPECT_EQ(del.back(), -1);
        break;
      case m::skewed:
        EXPECT_EQ(b.insert_offsets()[1] - b.insert_offsets()[0], 1100);  // >= 1,024
        break;
      case m::limits:
        EXPECT_EQ(del[1], std::numeric_limits<std::int64_t>::min());
        EXPECT_EQ(del[2], std::numeric_limits<std::int64_t>::max());
        EXPECT_EQ(del[3], bound);
        EXPECT_EQ(b.insert_members().front(), 0);
        break;
      case m::empty:
        EXPECT_TRUE(b.empty());
        break;
    }
    const auto s = h.apply(res_, b.view());
    h.check_integrity(res_);
    EXPECT_EQ(s.inserted_ids.size(), b.num_insertions());
    if (mix == m::reuse) {
      // Delete-then-reuse within the batch: under erase_first the new hyperedges fit deleted
      // slots (the batch's or earlier ones), so no id is appended.
      for (const std::int64_t id : s.inserted_ids) EXPECT_LT(id, bound) << id;
      EXPECT_EQ(h.hyperedge_id_bound(), bound);
    }
    if (mix == m::dead_ids) {
      EXPECT_EQ(s.ignored_deletions, 5);
    }
    if (mix == m::duplicates) {
      EXPECT_GT(s.dropped_members, 0);
    }
    if (mix == m::limits) {
      EXPECT_EQ(s.ignored_deletions, 3);
    }
    if (mix == m::delete_all) {
      EXPECT_EQ(h.num_hyperedges(), 0);
    }
  }
}

TEST_F(HyperedgeBatches, TheViewOverloadIsTheSameStream) {
  const hgraph h = start();
  const auto v = h.h2v();
  gen::hyperedge_batch_options o;
  o.mix = gen::hyperedge_batch_mix::reinsert;
  o.seed = 3;
  const auto a = gen::random_hyperedge_batch(res_, h, o);
  const auto b =
      gen::random_hyperedge_batch<std::int32_t, std::int32_t>(h.num_vertices(), v, h.weights(), o);
  EXPECT_EQ(hash_of(a), hash_of(b));
  o.size = -2;
  EXPECT_THROW((void)gen::random_hyperedge_batch(res_, h, o), dyng::invalid_argument_error);
  // An empty hypergraph without vertices: only empty batches.
  const ugraph none = ugraph::from_hyperedges(res_, dyng::hyperedge_list<std::int32_t>{}.view());
  gen::hyperedge_batch_options any;
  any.mix = gen::hyperedge_batch_mix::insert_only;
  EXPECT_TRUE(gen::random_hyperedge_batch(res_, none, any).empty());
}

// ---------------------------------------------------------------------------------------------
// generators::legacy: ESCHER-GPU
// ---------------------------------------------------------------------------------------------

namespace legacy = dyng::generators::legacy;

#if defined(__GLIBCXX__) && defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 11
TEST(LegacyStreams, TheStandardLibraryIsTheOriginalsOne) {
  EXPECT_TRUE(legacy::standard_library_matches_legacy_streams());
}
#endif

TEST(LegacyEscher, HyperedgeToVertexDrawsAsTheOriginal) {
  // hyperedge2vertex(8, 5, 1, 100) of ESCHER-GPU with std::mt19937_64(1): the loaded rows of
  // `main 8 5 1 100 4096` (cpp/tests/data/goldens/cbst_trace/escher_gpu_test/001-main.cbt, which
  // dyng_hypergraph_ids_tests compares), here re-derived from the standard library.
  std::mt19937_64 rng(1);
  legacy::escher_hypergraph_options o;
  o.num_hyperedges = 8;
  o.max_cardinality = 5;
  o.min_vertex = 0;
  o.max_vertex = 99;
  const auto l = legacy::escher_hypergraph<std::int32_t, dyng::unweighted>(rng, o);
  EXPECT_EQ(l.num_vertices, 100);
  EXPECT_EQ(l.num_hyperedges(), 8);
#if defined(__GLIBCXX__)
  std::mt19937_64 theirs(1);
  std::uniform_int_distribution<int> size(1, 5);
  std::uniform_int_distribution<int> vertex(1, 100);
  for (std::int64_t i = 0; i < 8; ++i) {
    const int card = std::min(size(theirs), 100);
    std::set<int> u;
    while (static_cast<int>(u.size()) < card) u.insert(vertex(theirs));
    std::vector<std::int32_t> want;
    for (const int v : u) want.push_back(v - 1);
    EXPECT_EQ(std::vector<std::int32_t>(l.members.begin() + l.offsets[i],
                                        l.members.begin() + l.offsets[i + 1]),
              want);
  }
  EXPECT_EQ(rng(), theirs());  // the same number of draws
#endif
  o.max_vertex = -1;
  EXPECT_THROW((void)(legacy::escher_hypergraph<std::int32_t, dyng::unweighted>(rng, o)),
               dyng::invalid_argument_error);
}

TEST(LegacyEscher, TheBatchGeneratorReadsTheLiveState) {
  const dyng::resources res = dyng::resources::sequential();
  std::mt19937_64 rng(7);
  legacy::escher_hypergraph_options g;
  g.num_hyperedges = 50;
  g.max_cardinality = 4;
  g.max_vertex = 29;
  const auto list = legacy::escher_hypergraph<std::int32_t, std::int32_t>(rng, g);
  for (const auto w : list.weights) EXPECT_EQ(w, 1);
  dyng::hypergraph_properties p;
  p.capacity = dyng::capacity_policy::escher_gpu();
  hgraph h = hgraph::from_hyperedges(res, list.view(), p);
  // The demo batch: 2 deletions, 3 insertions.
  legacy::escher_batch_generator demo(rng, legacy::escher_batch_options{});
  const auto b = demo.next(res, h);
  EXPECT_EQ(b.num_deletions(), 2u);
  EXPECT_EQ(b.num_insertions(), 3u);
  EXPECT_TRUE(std::is_sorted(b.delete_hyperedges().begin(), b.delete_hyperedges().end()));
  (void)h.apply(res, b.view());
  // --batch-size 7 in sample mode: 3 deletions, 4 copies of live hyperedges.
  legacy::escher_batch_generator sample(
      rng, legacy::escher_batch_options::batch_size(7, legacy::escher_insert_mode::sample, g));
  const auto c = sample.next(res, h);
  EXPECT_EQ(c.num_deletions(), 3u);
  EXPECT_EQ(c.num_insertions(), 4u);
  EXPECT_EQ(legacy::to_string(legacy::escher_insert_mode::sample), "sample");
  legacy::escher_batch_options bad;
  bad.num_deletions = 3;
  EXPECT_THROW(legacy::escher_batch_generator(rng, bad), dyng::invalid_argument_error);
}

}  // namespace
