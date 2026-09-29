// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_enum_batch_test.cpp
 * @brief generators::legacy::cycle_enum_batch() reproduces CycleEnumeration-GPU@0a976ad's
 *        generate_batch() bit for bit (the committed fixtures: several graphs, seeds, counts and
 *        locality windows, and the original's errors), the original's BatchGeneratorTest cases, and
 *        the library-owned reproductions of libstdc++'s uniform_int_distribution and shuffle for
 *        64-bit engines.
 *
 * Expected batches: cpp/tests/data/cycle_enum/generator, written by the pinned original
 * (parity/fixtures/cycle_enum/make_cycle_enum_fixtures.sh).
 */
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"
#include "util/rng.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using dyng::unweighted;
using dyng::test::data_path;
using dyng::test::read_text;
namespace legacy = dyng::generators::legacy;

using graph_t = dyng::graph<std::int32_t, std::int64_t, unweighted>;

graph_t load(const std::string& file) {
  const auto res = dyng::resources::sequential();
  const auto edges = dyng::io::read_edge_list<std::int32_t, unweighted>(file);
  return graph_t::from_edges(res, edges.view(), dyng::graph_properties::cycle_enum_compatible());
}

template <typename batch_t>
std::string text_of(const batch_t& b) {
  return dyng::test::batch_text(b.delete_src(), b.delete_dst(), b.insert_src(), b.insert_dst());
}

#if defined(__GLIBCXX__)
TEST(LegacyRng, UniformInt64MatchesLibstdcxx) {
  const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {
      {0, 0},
      {0, 1},
      {0, 2},
      {5, 9},
      {0, 299},
      {17, 1000000},
      {0, (1ULL << 40) + 3},
      {1, (1ULL << 63) + 12345},
      {0, ~0ULL - 1},
      {0, ~0ULL},
  };
  for (const auto& [a, b] : ranges) {
    SCOPED_TRACE(b);
    std::mt19937_64 ours(99);
    std::mt19937_64 theirs(99);
    std::uniform_int_distribution<std::size_t> dist(a, b);
    for (int i = 0; i < 2000; ++i) {
      ASSERT_EQ(dyng::detail::legacy_uniform_int64(ours, a, b), dist(theirs)) << i;
    }
  }
}

TEST(LegacyRng, ShuffleMatchesLibstdcxx) {
  for (const std::size_t size : {0U, 1U, 2U, 3U, 4U, 5U, 10U, 11U, 64U, 255U, 1000U, 100001U}) {
    SCOPED_TRACE(size);
    std::vector<std::uint32_t> ours(size);
    std::iota(ours.begin(), ours.end(), 0U);
    std::vector<std::uint32_t> theirs = ours;
    std::mt19937_64 a(size + 7);
    std::mt19937_64 b(size + 7);
    dyng::detail::legacy_shuffle(ours, a);
    std::shuffle(theirs.begin(), theirs.end(), b);
    EXPECT_EQ(ours, theirs);
    EXPECT_EQ(a(), b());  // the same number of draws
  }
}
#endif

// Every case of cases.txt against the original's output ("error" = its std::invalid_argument).
TEST(CycleEnumBatch, FixturesMatchGenerateBatch) {
  std::istringstream cases(read_text(data_path("cycle_enum/generator/cases.txt")));
  std::string line;
  int index = 0;
  while (std::getline(cases, line)) {
    std::istringstream fields(line);
    std::string file;
    std::int64_t deletions = 0;
    std::int64_t insertions = 0;
    std::uint64_t seed = 0;
    legacy::cycle_enum_batch_options opt;
    fields >> file >> deletions >> insertions >> seed;
    std::int64_t window = -1;
    if (fields >> window) {
      opt.locality_window = window;
    }
    opt.num_deletions = deletions;
    opt.num_insertions = insertions;
    opt.seed = seed;
    char name[16];
    std::snprintf(name, sizeof(name), "g%02d", index++);
    SCOPED_TRACE(std::string(name) + ": " + line);
    const std::string expected =
        read_text(data_path(std::string("cycle_enum/generator/") + name + ".expected"));
    const graph_t g = load(data_path("cycle_enum/generator/" + file));
    if (expected == "error\n") {
      EXPECT_THROW((void)legacy::cycle_enum_batch(g.view().out, opt), dyng::invalid_argument_error);
      continue;
    }
    const auto batch = legacy::cycle_enum_batch(g.view().out, opt);
    EXPECT_EQ(text_of(batch), expected);
    // The other graph types give the same batch.
    const auto edges = dyng::io::read_edge_list<std::int64_t, std::int32_t>(
        data_path("cycle_enum/generator/" + file));
    const auto w = dyng::graph<std::int64_t, std::int64_t, std::int32_t>::from_edges(
        dyng::resources::sequential(), edges.view(),
        dyng::graph_properties::cycle_enum_compatible());
    EXPECT_EQ(text_of(legacy::cycle_enum_batch(w.view().out, opt)), expected);
  }
  EXPECT_EQ(index, 22);
}

graph_t ring(std::int32_t n) {
  dyng::edge_list<std::int32_t, unweighted> e;
  e.num_vertices = n;
  for (std::int32_t v = 0; v < n; ++v) {
    e.add_edge(v, (v + 1) % n);
  }
  return graph_t::from_edges(dyng::resources::sequential(), e.view(),
                             dyng::graph_properties::cycle_enum_compatible());
}

// The original's BatchGeneratorTest cases.
TEST(CycleEnumBatch, HonorsRequestedCountsAndIsDeterministic) {
  const graph_t g = ring(12);
  legacy::cycle_enum_batch_options opt;
  opt.num_deletions = 3;
  opt.num_insertions = 4;
  opt.seed = 42;
  const auto a = legacy::cycle_enum_batch(g.view().out, opt);
  EXPECT_EQ(a.num_deletions(), 3u);
  EXPECT_EQ(a.num_insertions(), 4u);
  EXPECT_EQ(a.num_weights(), 0);
  const auto b = legacy::cycle_enum_batch(g.view().out, opt);
  EXPECT_EQ(text_of(a), text_of(b));
}

TEST(CycleEnumBatch, DeletionsExistInsertionsDoNot) {
  const graph_t g = ring(12);
  legacy::cycle_enum_batch_options opt;
  opt.num_deletions = 4;
  opt.num_insertions = 5;
  opt.seed = 99;
  const auto batch = legacy::cycle_enum_batch(g.view().out, opt);
  for (std::size_t i = 0; i < batch.num_deletions(); ++i) {
    EXPECT_EQ(batch.delete_dst()[i], (batch.delete_src()[i] + 1) % 12);
  }
  for (std::size_t i = 0; i < batch.num_insertions(); ++i) {
    EXPECT_NE(batch.insert_src()[i], batch.insert_dst()[i]);
    EXPECT_NE(batch.insert_dst()[i], (batch.insert_src()[i] + 1) % 12);
  }
}

TEST(CycleEnumBatch, LocalityWindowConfinesEndpoints) {
  const graph_t g = ring(40);
  legacy::cycle_enum_batch_options opt;
  opt.num_deletions = 2;
  opt.num_insertions = 3;
  opt.seed = 5;
  opt.locality_window = 6;
  const auto batch = legacy::cycle_enum_batch(g.view().out, opt);
  std::int32_t lo = 1000;
  std::int32_t hi = -1;
  for (const auto* list :
       {&batch.delete_src(), &batch.delete_dst(), &batch.insert_src(), &batch.insert_dst()}) {
    for (const std::int32_t v : *list) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  EXPECT_LT(hi - lo, 6);
}

TEST(CycleEnumBatch, Errors) {
  const graph_t g = ring(8);  // only 8 ring edges
  legacy::cycle_enum_batch_options opt;
  opt.num_deletions = 100;
  opt.seed = 1;
  EXPECT_THROW((void)legacy::cycle_enum_batch(g.view().out, opt), dyng::invalid_argument_error);
  opt.num_deletions = -1;
  EXPECT_THROW((void)legacy::cycle_enum_batch(g.view().out, opt), dyng::invalid_argument_error);
  opt.num_deletions = 0;
  EXPECT_EQ(legacy::cycle_enum_batch(g.view().out, opt).empty(), true);  // nothing requested
  const graph_t one = ring(1);
  opt.num_insertions = 1;
  EXPECT_THROW((void)legacy::cycle_enum_batch(one.view().out, opt), dyng::invalid_argument_error);
  // Unsorted rows (a MOSP-order graph) are rejected.
  using mosp_graph = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 3;
  e.num_weights = 1;
  e.add_edge(0, 2, {1});
  e.add_edge(0, 1, {1});
  const auto m = mosp_graph::from_edges(dyng::resources::sequential(), e.view(),
                                        dyng::graph_properties::mosp_compatible());
  EXPECT_THROW((void)legacy::cycle_enum_batch(m.view().out, opt), dyng::invalid_argument_error);
}

TEST(CycleEnumBatch, WeightedGraphInsertionsWeighOne) {
  using wgraph = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 20;
  e.num_weights = 2;
  for (std::int32_t v = 0; v < 20; ++v) {
    e.add_edge(v, (v + 1) % 20, {3, 4});
  }
  const auto res = dyng::resources::sequential();
  auto g = wgraph::from_edges(res, e.view(), dyng::graph_properties::cycle_enum_compatible());
  legacy::cycle_enum_batch_options opt;
  opt.num_deletions = 3;
  opt.num_insertions = 5;
  opt.seed = 3;
  const auto batch = legacy::cycle_enum_batch(g.view().out, opt);
  EXPECT_EQ(batch.num_weights(), 2);
  EXPECT_EQ(batch.insert_weights(), std::vector<std::int32_t>(10, 1));
  // The same pairs as on the unweighted ring.
  EXPECT_EQ(text_of(batch), text_of(legacy::cycle_enum_batch(ring(20).view().out, opt)));
  EXPECT_NO_THROW((void)g.apply(res, batch.view()));
  EXPECT_EQ(g.num_edges(), 22);
}

}  // namespace
