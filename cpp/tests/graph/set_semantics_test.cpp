// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file set_semantics_test.cpp
 * @brief graph_properties::cycle_enum_compatible() and batch_semantics::set(): the unweighted
 *        graph, Step 0 (the normalized batch equals CycleEnumeration-GPU's prepare_batch), the
 *        sorted-row apply (byte-equal CSR to its apply_batch on the committed fixtures and on
 *        randomized batches against an independent edge-set model), the switches and their errors,
 *        and the parallel assembly.
 *
 * The fixtures in cpp/tests/data/cycle_enum/cases come from the pinned original
 * (parity/fixtures/cycle_enum/make_cycle_enum_fixtures.sh): 80 random graphs with arbitrary
 * batches (absent deletions, present insertions, self-loops, duplicates, delete-then-reinsert
 * pairs, new vertices), each with the original's prepare_batch() and apply_batch() output.
 */
#include "graph/graph_impl.hpp"
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using dyng::batch_semantics;
using dyng::graph_properties;
using dyng::unweighted;
using dyng::test::data_path;

TEST(SetSemantics, Presets) {
  constexpr batch_semantics s = batch_semantics::set();
  EXPECT_TRUE(s.as_sets);
  EXPECT_EQ(s.on_existing_insert, batch_semantics::existing_insert::ignore);
  EXPECT_EQ(s.on_missing_delete, batch_semantics::missing_delete::ignore);
  EXPECT_EQ(s.on_self_loop, batch_semantics::self_loop::drop);
  EXPECT_TRUE(s.deletions_first);
  EXPECT_TRUE(s.allow_vertex_growth);
  EXPECT_FALSE(batch_semantics::upsert_last_wins().as_sets);
  EXPECT_FALSE(batch_semantics{}.as_sets);

  constexpr graph_properties p = graph_properties::cycle_enum_compatible();
  EXPECT_TRUE(p.directed);
  EXPECT_TRUE(p.store_transposed);
  EXPECT_EQ(p.num_weights, 0);
  EXPECT_EQ(p.layout, dyng::row_layout::compact);
  EXPECT_EQ(p.order, dyng::row_order::sorted);
  EXPECT_EQ(p.parallel_edges, dyng::multi_edges::forbid);
  EXPECT_TRUE(p.semantics.as_sets);
}

TEST(SetSemantics, UnweightedTypes) {
  EXPECT_TRUE(dyng::is_unweighted_v<unweighted>);
  EXPECT_FALSE(dyng::is_unweighted_v<std::int32_t>);
  EXPECT_TRUE(unweighted{} == unweighted{});
  EXPECT_FALSE(unweighted{} != unweighted{});
  EXPECT_FALSE(unweighted{} < unweighted{});
  EXPECT_EQ((dyng::edge_batch<std::int32_t, unweighted>().num_weights()), 0);
  EXPECT_EQ((dyng::edge_batch<std::int32_t, std::int32_t>().num_weights()), 1);
  EXPECT_EQ((dyng::edge_batch_view<std::int32_t, unweighted>{}.num_weights), 0);
  EXPECT_THROW((dyng::edge_batch<std::int32_t, unweighted>(1)), dyng::invalid_argument_error);
  dyng::edge_batch<std::int32_t, unweighted> b;
  b.insert_edge(0, 1);
  EXPECT_EQ(b.num_insertions(), 1u);
  EXPECT_TRUE(b.insert_weights().empty());
  dyng::edge_batch<std::int32_t, std::int32_t> w;
  EXPECT_THROW(w.insert_edge(0, 1), dyng::invalid_argument_error);  // it needs one weight
}

// ---------------------------------------------------------------------------------------------
// Typed over the graph types a cycle_enum_compatible() graph is used with
// ---------------------------------------------------------------------------------------------

template <typename graph_t>
class SetApply : public ::testing::Test {
 public:
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  using batch_t = dyng::edge_batch<vertex_t, weight_t>;

  /// A graph on n vertices with the given edges, no weight columns, cycle_enum_compatible().
  static graph_t build(const dyng::resources& res, std::int64_t n,
                       const std::vector<std::pair<std::int64_t, std::int64_t>>& edges,
                       graph_properties props = graph_properties::cycle_enum_compatible()) {
    dyng::edge_list<vertex_t, weight_t> list;
    list.num_vertices = static_cast<vertex_t>(n);
    for (const auto& [u, v] : edges) {
      list.add_edge(static_cast<vertex_t>(u), static_cast<vertex_t>(v));
    }
    return graph_t::from_edges(res, list.view(), props);
  }

  static bool has_edge(const graph_t& g, std::int64_t u, std::int64_t v) {
    const auto out = g.view().out;
    if (u >= out.num_vertices()) {
      return false;
    }
    return std::binary_search(out.col_ind.begin() + out.row_ptr[static_cast<std::size_t>(u)],
                              out.col_ind.begin() + out.row_ptr[static_cast<std::size_t>(u) + 1],
                              static_cast<vertex_t>(v));
  }

  static std::vector<std::pair<std::int64_t, std::int64_t>> edges_of(const graph_t& g) {
    const auto out = g.view().out;
    std::vector<std::pair<std::int64_t, std::int64_t>> list;
    for (std::int64_t u = 0; u < out.num_vertices(); ++u) {
      for (auto e = out.row_ptr[static_cast<std::size_t>(u)];
           e < out.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
        list.emplace_back(u, out.col_ind[static_cast<std::size_t>(e)]);
      }
    }
    return list;
  }

  /// Applies `b` and returns the normalized batch as text ("- u v" then "+ u v").
  static std::string apply(const dyng::resources& res, graph_t& g, const batch_t& b,
                           dyng::apply_summary* summary = nullptr) {
    dyng::detail::apply_delta<vertex_t> delta;
    const auto s = dyng::detail::graph_access::apply(res, g, b.view(), &delta);
    if (summary != nullptr) {
      *summary = s;
    }
    EXPECT_EQ(delta.weight_increased.size(), delta.insert_src.size() * g.num_weights());
    EXPECT_TRUE(std::all_of(delta.weight_increased.begin(), delta.weight_increased.end(),
                            [](std::uint8_t f) { return f == 0; }));
    g.check_integrity(res);
    return dyng::test::batch_text(delta.delete_src, delta.delete_dst, delta.insert_src,
                                  delta.insert_dst);
  }
};

using graph_types = ::testing::Types<dyng::graph<std::int32_t, std::int32_t, unweighted>,
                                     dyng::graph<std::int32_t, std::int64_t, unweighted>,
                                     dyng::graph<std::int32_t, std::int32_t, std::int32_t>,
                                     dyng::graph<std::int64_t, std::int64_t, std::int32_t>>;
TYPED_TEST_SUITE(SetApply, graph_types, dyng::test::type_index_name);

// The original's DirectedGraphTest cases.
TYPED_TEST(SetApply, BuildAndHasEdge) {
  const auto res = dyng::resources::sequential();
  const auto g = TestFixture::build(res, 3, {{0, 1}, {1, 2}, {2, 0}});
  EXPECT_EQ(g.num_vertices(), 3);
  EXPECT_EQ(g.num_weights(), 0);
  EXPECT_TRUE(TestFixture::has_edge(g, 0, 1));
  EXPECT_TRUE(TestFixture::has_edge(g, 2, 0));
  EXPECT_FALSE(TestFixture::has_edge(g, 1, 0));
  EXPECT_FALSE(TestFixture::has_edge(g, 0, 2));
}

TYPED_TEST(SetApply, ApplyBatchAddsAndRemovesEdges) {
  const auto res = dyng::resources::sequential();
  auto g = TestFixture::build(res, 3, {{0, 1}, {1, 2}, {2, 0}});
  typename TestFixture::batch_t b(0);
  b.delete_edge(1, 2);
  b.insert_edge(0, 2);
  EXPECT_EQ(TestFixture::apply(res, g, b), "- 1 2\n+ 0 2\n");
  EXPECT_FALSE(TestFixture::has_edge(g, 1, 2));
  EXPECT_TRUE(TestFixture::has_edge(g, 0, 2));
  EXPECT_TRUE(TestFixture::has_edge(g, 0, 1));
  EXPECT_TRUE(TestFixture::has_edge(g, 2, 0));
  EXPECT_EQ(g.version(), 1u);
}

// The original's UpdateBatchValidationTest cases: every change that does not alter the graph
// leaves the normalized batch; a delete-then-reinsert pair stays in both lists.
TYPED_TEST(SetApply, NormalizedBatchDropsNoOps) {
  const auto res = dyng::resources::sequential();
  struct change_case {
    const char* name;
    std::vector<std::pair<int, int>> deletions;
    std::vector<std::pair<int, int>> insertions;
    std::string prepared;
    std::int64_t vertices_after;
    std::int64_t self_loops;
    std::int64_t cancelled;
  };
  const std::vector<change_case> cases = {
      {"delete missing edge", {{2, 1}}, {}, "", 3, 0, 0},
      {"insert existing edge", {}, {{0, 1}}, "", 3, 0, 0},
      {"insert self-loop", {}, {{1, 1}}, "", 3, 1, 0},
      {"delete self-loop", {{1, 1}}, {}, "", 3, 1, 0},
      {"delete and reinsert", {{0, 1}}, {{0, 1}}, "- 0 1\n+ 0 1\n", 3, 0, 1},
      {"duplicate changes", {{1, 0}, {1, 0}}, {{0, 2}, {0, 2}}, "- 1 0\n+ 0 2\n", 3, 0, 0},
      {"insert through new vertices", {}, {{2, 4}, {4, 0}}, "+ 2 4\n+ 4 0\n", 5, 0, 0},
      {"mixed",
       {{2, 1}, {1, 0}, {3, 3}},
       {{0, 1}, {2, 5}, {5, 1}, {5, 5}},
       "- 1 0\n+ 2 5\n+ 5 1\n",
       6,
       2,
       0},
  };
  for (const change_case& c : cases) {
    SCOPED_TRACE(c.name);
    // 0->1, 1->2, 2->0, 1->0: one 2-cycle and one 3-cycle.
    auto g = TestFixture::build(res, 3, {{0, 1}, {1, 2}, {2, 0}, {1, 0}});
    typename TestFixture::batch_t b(0);
    std::set<std::pair<std::int64_t, std::int64_t>> model = {{0, 1}, {1, 2}, {2, 0}, {1, 0}};
    for (const auto& [u, v] : c.deletions) {
      b.delete_edge(u, v);
      model.erase({u, v});
    }
    for (const auto& [u, v] : c.insertions) {
      b.insert_edge(u, v);
      if (u != v) {
        model.emplace(u, v);
      }
    }
    dyng::apply_summary s;
    EXPECT_EQ(TestFixture::apply(res, g, b, &s), c.prepared);
    EXPECT_EQ(g.num_vertices(), c.vertices_after);
    EXPECT_EQ(s.num_vertices_after, c.vertices_after);
    EXPECT_EQ(s.inserted_vertices, c.vertices_after - 3);
    EXPECT_EQ(s.dropped_self_loops, c.self_loops);
    EXPECT_EQ(s.cancelled_pairs, c.cancelled);
    EXPECT_EQ(s.updated_edges, 0);
    const auto edges = TestFixture::edges_of(g);
    const std::set<std::pair<std::int64_t, std::int64_t>> got(edges.begin(), edges.end());
    EXPECT_EQ(got, model);
    EXPECT_EQ(static_cast<std::int64_t>(edges.size()), 4 - s.deleted_edges + s.inserted_edges);
  }
}

TYPED_TEST(SetApply, SummaryCounts) {
  const auto res = dyng::resources::sequential();
  auto g = TestFixture::build(res, 4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  typename TestFixture::batch_t b(0);
  b.delete_edge(0, 1);  // deleted
  b.delete_edge(0, 1);  // a duplicate: merged, not counted
  b.delete_edge(1, 0);  // missing
  b.delete_edge(2, 2);  // self-loop
  b.insert_edge(1, 2);  // existing: ignored
  b.insert_edge(0, 1);  // re-inserted: a cancelled pair
  b.insert_edge(0, 2);  // new
  b.insert_edge(0, 2);  // a duplicate
  b.insert_edge(3, 3);  // self-loop
  dyng::apply_summary s;
  EXPECT_EQ(TestFixture::apply(res, g, b, &s), "- 0 1\n+ 0 1\n+ 0 2\n");
  EXPECT_EQ(s.deleted_edges, 1);
  EXPECT_EQ(s.ignored_deletions, 1);
  EXPECT_EQ(s.inserted_edges, 2);
  EXPECT_EQ(s.ignored_insertions, 1);
  EXPECT_EQ(s.cancelled_pairs, 1);
  EXPECT_EQ(s.dropped_self_loops, 2);
  EXPECT_EQ(s.updated_edges, 0);
  EXPECT_EQ(g.num_edges(), 5);
}

TYPED_TEST(SetApply, SwitchesAndErrors) {
  const auto res = dyng::resources::sequential();
  const std::vector<std::pair<std::int64_t, std::int64_t>> base = {{0, 1}, {1, 2}, {2, 0}};
  using batch_t = typename TestFixture::batch_t;
  {
    graph_properties p = graph_properties::cycle_enum_compatible();
    p.semantics.on_missing_delete = batch_semantics::missing_delete::error;
    auto g = TestFixture::build(res, 3, base, p);
    batch_t b(0);
    b.delete_edge(1, 0);
    EXPECT_THROW(g.apply(res, b.view()), dyng::invalid_argument_error);
    batch_t ok(0);
    ok.delete_edge(0, 1);
    ok.delete_edge(0, 1);  // a duplicate of an existing edge is merged, not an error
    EXPECT_NO_THROW(g.apply(res, ok.view()));
  }
  {
    graph_properties p = graph_properties::cycle_enum_compatible();
    p.semantics.on_existing_insert = batch_semantics::existing_insert::error;
    auto g = TestFixture::build(res, 3, base, p);
    batch_t b(0);
    b.insert_edge(0, 1);
    EXPECT_THROW(g.apply(res, b.view()), dyng::invalid_argument_error);
    batch_t ok(0);  // deleted and re-inserted: not an existing edge at insertion time
    ok.delete_edge(0, 1);
    ok.insert_edge(0, 1);
    EXPECT_NO_THROW(g.apply(res, ok.view()));
  }
  {
    graph_properties p = graph_properties::cycle_enum_compatible();
    p.semantics.on_self_loop = batch_semantics::self_loop::error;
    auto g = TestFixture::build(res, 3, base, p);
    batch_t b(0);
    b.insert_edge(1, 1);
    EXPECT_THROW(g.apply(res, b.view()), dyng::invalid_argument_error);
  }
  {
    graph_properties p = graph_properties::cycle_enum_compatible();
    p.semantics.on_self_loop = batch_semantics::self_loop::keep;
    auto g = TestFixture::build(res, 3, base, p);
    batch_t b(0);
    b.insert_edge(1, 1);
    b.insert_edge(1, 1);
    dyng::apply_summary s;
    EXPECT_EQ(TestFixture::apply(res, g, b, &s), "+ 1 1\n");
    EXPECT_TRUE(TestFixture::has_edge(g, 1, 1));
    batch_t d(0);
    d.delete_edge(1, 1);
    EXPECT_EQ(TestFixture::apply(res, g, d), "- 1 1\n");
    EXPECT_FALSE(TestFixture::has_edge(g, 1, 1));
  }
  {
    graph_properties p = graph_properties::cycle_enum_compatible();
    p.semantics.allow_vertex_growth = false;
    auto g = TestFixture::build(res, 3, base, p);
    batch_t b(0);
    b.insert_edge(1, 3);
    EXPECT_THROW(g.apply(res, b.view()), dyng::invalid_argument_error);
  }
  {
    auto g = TestFixture::build(res, 3, base);
    batch_t b(0);
    b.insert_edge(-1, 2);
    EXPECT_THROW(g.apply(res, b.view()), dyng::invalid_argument_error);
    batch_t d(0);
    d.delete_edge(0, -2);
    EXPECT_THROW(g.apply(res, d.view()), dyng::invalid_argument_error);
    EXPECT_EQ(g.version(), 0u);
  }
  for (const auto change : {0, 1, 2, 3}) {
    graph_properties p = graph_properties::cycle_enum_compatible();
    switch (change) {
      case 0:
        p.semantics.on_existing_insert = batch_semantics::existing_insert::upsert;
        break;
      case 1:
        p.semantics.deletions_first = false;
        break;
      case 2:
        p.order = dyng::row_order::append;
        break;
      default:
        p.parallel_edges = dyng::multi_edges::allow;
        break;
    }
    SCOPED_TRACE(change);
    auto g = TestFixture::build(res, 3, base, p);
    batch_t b(0);
    b.insert_edge(0, 2);
    EXPECT_THROW(g.apply(res, b.view()), dyng::not_supported_error);
  }
}

TYPED_TEST(SetApply, UndirectedGraphChangesBothDirections) {
  const auto res = dyng::resources::sequential();
  graph_properties p = graph_properties::cycle_enum_compatible();
  p.directed = false;
  auto g = TestFixture::build(res, 4, {{0, 1}, {1, 2}}, p);
  EXPECT_EQ(g.num_edges(), 4);
  typename TestFixture::batch_t b(0);
  b.delete_edge(1, 0);
  b.insert_edge(2, 3);
  b.insert_edge(3, 2);  // the same undirected edge: merged
  EXPECT_EQ(TestFixture::apply(res, g, b), "- 0 1\n- 1 0\n+ 2 3\n+ 3 2\n");
  const auto edges = TestFixture::edges_of(g);
  EXPECT_EQ(edges,
            (std::vector<std::pair<std::int64_t, std::int64_t>>{{1, 2}, {2, 1}, {2, 3}, {3, 2}}));
}

// The original's ApplyBatchMatchesSetSemanticsOnRandomBatches, on every host backend.
TYPED_TEST(SetApply, RandomBatchesMatchSetModel) {
  for (const dyng::backend backend : dyng::test::host_backends()) {
    const auto res = backend == dyng::backend::openmp ? dyng::resources::openmp(4)
                                                      : dyng::resources::sequential();
    std::mt19937_64 rng(314);
    for (int trial = 0; trial < 300; ++trial) {
      const std::int64_t n = 2 + static_cast<std::int64_t>(rng() % 12);
      std::set<std::pair<std::int64_t, std::int64_t>> edges;
      for (std::int64_t u = 0; u < n; ++u) {
        for (std::int64_t v = 0; v < n; ++v) {
          if (u != v && rng() % 3 == 0) {
            edges.emplace(u, v);
          }
        }
      }
      auto g = TestFixture::build(res, n, {edges.begin(), edges.end()});
      typename TestFixture::batch_t b(0);
      const auto vertex = [&] { return static_cast<std::int64_t>(rng() % (n + 3)); };
      for (std::size_t i = rng() % 8; i > 0; --i) {
        const auto u = vertex();
        const auto v = vertex();
        b.delete_edge(u, v);
      }
      for (std::size_t i = rng() % 8; i > 0; --i) {
        const auto u = vertex();
        const auto v = vertex();
        b.insert_edge(u, v);
      }
      std::int64_t expected_vertices = n;
      for (std::size_t j = 0; j < b.num_deletions(); ++j) {
        edges.erase({b.delete_src()[j], b.delete_dst()[j]});
      }
      for (std::size_t i = 0; i < b.num_insertions(); ++i) {
        const std::int64_t u = b.insert_src()[i];
        const std::int64_t v = b.insert_dst()[i];
        if (u != v) {
          edges.emplace(u, v);
          expected_vertices = std::max(expected_vertices, std::max(u, v) + 1);
        }
      }
      (void)TestFixture::apply(res, g, b);
      ASSERT_EQ(g.num_vertices(), expected_vertices) << trial;
      EXPECT_EQ(TestFixture::edges_of(g),
                (std::vector<std::pair<std::int64_t, std::int64_t>>(edges.begin(), edges.end())))
          << trial;
    }
  }
}

// Byte parity with the original's prepare_batch() + apply_batch() on the committed fixtures, and
// the edge-set recount of dyng::testing on the same cases.
TYPED_TEST(SetApply, FixturesMatchPrepareAndApplyBatch) {
  using vertex_t = typename TestFixture::vertex_t;
  const std::vector<std::string> names = dyng::test::cycle_enum_case_names();
  ASSERT_EQ(names.size(), 80u);
  for (const dyng::backend backend : dyng::test::host_backends()) {
    const auto res = backend == dyng::backend::openmp ? dyng::resources::openmp(4)
                                                      : dyng::resources::sequential();
    for (const std::string& name : names) {
      SCOPED_TRACE(name);
      const auto c =
          dyng::test::read_cycle_enum_case(data_path("cycle_enum/cases/" + name + ".txt"));
      const std::string expected =
          dyng::test::read_text(data_path("cycle_enum/cases/" + name + ".expected"));
      std::vector<std::pair<std::int64_t, std::int64_t>> edges;
      for (std::size_t i = 0; i < c.src.size(); ++i) {
        edges.emplace_back(c.src[i], c.dst[i]);
      }
      auto g = TestFixture::build(res, c.n, edges);
      const auto before = g.to_csr(res);
      typename TestFixture::batch_t b(0);
      for (std::size_t j = 0; j < c.del_src.size(); ++j) {
        b.delete_edge(static_cast<vertex_t>(c.del_src[j]), static_cast<vertex_t>(c.del_dst[j]));
      }
      for (std::size_t i = 0; i < c.ins_src.size(); ++i) {
        b.insert_edge(static_cast<vertex_t>(c.ins_src[i]), static_cast<vertex_t>(c.ins_dst[i]));
      }
      const std::string prepared = TestFixture::apply(res, g, b);
      EXPECT_EQ(prepared, dyng::test::section(expected, "prepared", "after\n"));
      const std::string after = dyng::test::csr_text(g.view().out);
      const std::string expected_after = dyng::test::section(expected, "after", "oracle_before");
      ASSERT_FALSE(expected_after.empty());
      EXPECT_EQ(after, expected_after);
      // The independent recount of dyng::testing gives the same graph.
      const auto recount = dyng::testing::edge_set_after_batch(before.view(), b.view());
      EXPECT_EQ(dyng::test::csr_text(recount.view()), after);
    }
  }
}

// The parallel assembly (more than 65536 rows, several threads) writes the same bytes as the
// sequential one and as the edge-set model.
TYPED_TEST(SetApply, ParallelAssemblyEqualsSequential) {
  if (!dyng::backend_available(dyng::backend::openmp)) {
    GTEST_SKIP() << "OpenMP backend not built";
  }
  using vertex_t = typename TestFixture::vertex_t;
  const std::int64_t n = 150000;
  std::mt19937_64 rng(7);
  std::set<std::pair<std::int64_t, std::int64_t>> edge_set;
  while (edge_set.size() < 400000) {
    const auto u = static_cast<std::int64_t>(rng() % n);
    const auto v = static_cast<std::int64_t>(rng() % n);
    if (u != v) {
      edge_set.emplace(u, v);
    }
  }
  const std::vector<std::pair<std::int64_t, std::int64_t>> edges(edge_set.begin(), edge_set.end());
  typename TestFixture::batch_t b(0);
  for (int i = 0; i < 30000; ++i) {
    const auto& e = edges[rng() % edges.size()];
    b.delete_edge(static_cast<vertex_t>(e.first), static_cast<vertex_t>(e.second));
  }
  for (int i = 0; i < 30000; ++i) {
    const auto u = static_cast<vertex_t>(rng() % (n + 100));
    const auto v = static_cast<vertex_t>(rng() % (n + 100));
    b.insert_edge(u, v);
  }
  const auto seq = dyng::resources::sequential();
  auto g1 = TestFixture::build(seq, n, edges);
  const auto before = g1.to_csr(seq);
  const std::string batch1 = TestFixture::apply(seq, g1, b);
  const auto omp = dyng::resources::openmp(8);
  auto g2 = TestFixture::build(omp, n, edges);
  const std::string batch2 = TestFixture::apply(omp, g2, b);
  EXPECT_EQ(batch1, batch2);
  const std::string csr1 = dyng::test::csr_text(g1.view().out);
  EXPECT_EQ(csr1, dyng::test::csr_text(g2.view().out));
  EXPECT_EQ(csr1, dyng::test::csr_text(
                      dyng::testing::edge_set_after_batch(before.view(), b.view()).view()));
}

// Weights under set semantics: kept edges keep theirs, a new edge takes the weights of its first
// insertion in batch order, a deleted and re-inserted edge takes the new weights.
TEST(SetSemantics, WeightedGraphWeights) {
  using graph_t = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
  const auto res = dyng::resources::sequential();
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 3;
  list.num_weights = 2;
  list.add_edge(0, 1, {5, 50});
  list.add_edge(1, 2, {6, 60});
  list.add_edge(2, 0, {7, 70});
  auto g = graph_t::from_edges(res, list.view(), graph_properties::cycle_enum_compatible());
  EXPECT_EQ(g.num_weights(), 2);
  dyng::edge_batch<std::int32_t, std::int32_t> b(2);
  b.insert_edge(0, 2, {1, 10});  // new: the first insertion's weights
  b.insert_edge(0, 2, {2, 20});
  b.insert_edge(1, 2, {3, 30});  // existing, not deleted: ignored
  b.delete_edge(2, 0);
  b.insert_edge(2, 0, {4, 40});  // deleted and re-inserted: the new weights
  (void)g.apply(res, b.view());
  const auto out = g.to_csr(res);
  EXPECT_EQ(out.row_ptr, (std::vector<std::int64_t>{0, 2, 3, 4}));
  EXPECT_EQ(out.col_ind, (std::vector<std::int32_t>{1, 2, 2, 0}));
  EXPECT_EQ(out.weights, (std::vector<std::int32_t>{5, 1, 6, 4, 50, 10, 60, 40}));
}

TEST(SetSemantics, UnweightedInputsRejectWeights) {
  using graph_t = dyng::graph<std::int32_t, std::int32_t, unweighted>;
  const auto res = dyng::resources::sequential();
  dyng::edge_list<std::int32_t, unweighted> list;
  list.num_vertices = 2;
  list.num_weights = 1;
  list.src = {0};
  list.dst = {1};
  list.weights = {unweighted{}};
  EXPECT_THROW(
      (void)graph_t::from_edges(res, list.view(), graph_properties::cycle_enum_compatible()),
      dyng::invalid_argument_error);
  dyng::csr<std::int32_t, std::int32_t, unweighted> csr;
  csr.row_ptr = {0, 1, 1};
  csr.col_ind = {1};
  csr.weights = {unweighted{}};
  csr.num_weights = 1;
  EXPECT_THROW((void)graph_t::from_csr(res, csr.view(), graph_properties::cycle_enum_compatible()),
               dyng::invalid_argument_error);
  // A graph from an unsorted edge list with duplicates and self-loops is the sorted simple graph.
  dyng::edge_list<std::int32_t, unweighted> messy;
  messy.num_vertices = 3;
  for (const auto& [u, v] :
       std::vector<std::pair<int, int>>{{2, 0}, {0, 2}, {0, 1}, {0, 2}, {1, 1}}) {
    messy.add_edge(u, v);
  }
  const auto g = graph_t::from_edges(res, messy.view(), graph_properties::cycle_enum_compatible());
  const auto out = g.to_csr(res);
  EXPECT_EQ(out.row_ptr, (std::vector<std::int32_t>{0, 2, 2, 3}));
  EXPECT_EQ(out.col_ind, (std::vector<std::int32_t>{1, 2, 0}));
  g.check_integrity(res);
}

}  // namespace
