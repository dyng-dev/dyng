// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file structural_change_test.cpp
 * @brief compute_structural_change(): Step 0 on G_t. Under batch_semantics::set() it equals
 *        CycleEnumeration-GPU@0a976ad's prepare_batch() (the committed fixtures) and the normalized
 *        batch of graph::apply(); under every other semantics it is the change of the edge set
 *        that graph::apply() makes.
 */
#include "graph/structural_change.hpp"

#include "graph/graph_impl.hpp"
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using dyng::unweighted;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using graph_w = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using change_t = dyng::detail::structural_change<std::int32_t>;
using edge_set = std::set<std::pair<std::int32_t, std::int32_t>>;

template <typename list_t>
std::string list_text(const list_t& list, char tag) {
  std::string out;
  for (const auto& c : list) {
    out += tag;
    out += ' ' + std::to_string(c.source) + ' ' + std::to_string(c.target) + '\n';
  }
  return out;
}

template <typename graph_t>
edge_set edges_of(const graph_t& g) {
  const auto out = dyng::detail::graph_access::out_view(g);
  edge_set s;
  for (std::int32_t u = 0; u < out.num_vertices(); ++u) {
    for (auto e = out.row_ptr[static_cast<std::size_t>(u)];
         e < out.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
      if (out.col_ind[static_cast<std::size_t>(e)] != u) {
        s.emplace(u, out.col_ind[static_cast<std::size_t>(e)]);
      }
    }
  }
  return s;
}

TEST(StructuralChange, EqualsPrepareBatchOnTheFixtures) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::string> names = dyng::test::cycle_enum_case_names();
  ASSERT_EQ(names.size(), 80U);
  for (const std::string& name : names) {
    SCOPED_TRACE(name);
    const auto c = dyng::test::read_cycle_enum_case(
        dyng::test::data_path("cycle_enum/cases/" + name + ".txt"));
    const std::string expected =
        dyng::test::read_text(dyng::test::data_path("cycle_enum/cases/" + name + ".expected"));
    dyng::edge_list<std::int32_t, unweighted> list;
    list.num_vertices = static_cast<std::int32_t>(c.n);
    for (std::size_t i = 0; i < c.src.size(); ++i) {
      list.add_edge(static_cast<std::int32_t>(c.src[i]), static_cast<std::int32_t>(c.dst[i]));
    }
    graph_u g =
        graph_u::from_edges(res, list.view(), dyng::graph_properties::cycle_enum_compatible());
    dyng::edge_batch<std::int32_t, unweighted> batch;
    for (std::size_t i = 0; i < c.del_src.size(); ++i) {
      batch.delete_edge(static_cast<std::int32_t>(c.del_src[i]),
                        static_cast<std::int32_t>(c.del_dst[i]));
    }
    for (std::size_t i = 0; i < c.ins_src.size(); ++i) {
      batch.insert_edge(static_cast<std::int32_t>(c.ins_src[i]),
                        static_cast<std::int32_t>(c.ins_dst[i]));
    }
    change_t change;
    dyng::detail::compute_structural_change(dyng::detail::graph_access::out_view(g), batch.view(),
                                            g.properties(), change);
    EXPECT_EQ(list_text(change.deletions, '-') + list_text(change.insertions, '+'),
              dyng::test::section(expected, "prepared", "after"));
    dyng::detail::apply_delta<std::int32_t> delta;
    (void)dyng::detail::graph_access::apply(res, g, batch.view(), &delta);
    ASSERT_EQ(change.deletions.size(), delta.delete_src.size());
    ASSERT_EQ(change.insertions.size(), delta.insert_src.size());
    for (std::size_t i = 0; i < change.deletions.size(); ++i) {
      EXPECT_EQ(change.deletions[i].source, delta.delete_src[i]);
      EXPECT_EQ(change.deletions[i].target, delta.delete_dst[i]);
    }
    for (std::size_t i = 0; i < change.insertions.size(); ++i) {
      EXPECT_EQ(change.insertions[i].source, delta.insert_src[i]);
      EXPECT_EQ(change.insertions[i].target, delta.insert_dst[i]);
    }
  }
}

TEST(StructuralChange, IsTheEdgeSetChangeUnderEverySemantics) {
  const dyng::resources res = dyng::resources::sequential();
  std::vector<std::pair<const char*, dyng::graph_properties>> variants;
  variants.emplace_back("set", dyng::graph_properties::cycle_enum_compatible());
  variants.emplace_back("upsert", dyng::graph_properties{});
  {
    dyng::graph_properties p;
    p.semantics.on_existing_insert = dyng::batch_semantics::existing_insert::ignore;
    p.semantics.on_self_loop = dyng::batch_semantics::self_loop::keep;
    variants.emplace_back("ignore, keep self-loops", p);
  }
  {
    dyng::graph_properties p;
    p.semantics.deletions_first = false;
    variants.emplace_back("insertions first", p);
  }
  {
    dyng::graph_properties p = dyng::graph_properties::cycle_enum_compatible();
    p.directed = false;
    variants.emplace_back("undirected set", p);
  }
  {
    dyng::graph_properties p;
    p.directed = false;
    variants.emplace_back("undirected upsert", p);
  }
  std::mt19937_64 rng(2718);
  for (const auto& [name, props] : variants) {
    for (int trial = 0; trial < 60; ++trial) {
      const auto n = static_cast<std::int32_t>(2 + rng() % 9);
      std::uniform_int_distribution<std::int32_t> vertex(0, n + 1);
      dyng::edge_list<std::int32_t, std::int32_t> list;
      list.num_vertices = n;
      list.num_weights = 1;
      for (std::uint64_t i = rng() % (3 * static_cast<std::uint64_t>(n)); i > 0; --i) {
        const std::int32_t u = vertex(rng) % n;
        const std::int32_t v = vertex(rng) % n;
        list.add_edge(u, v, {1});
        if (!props.directed) {
          list.add_edge(v, u, {1});
        }
      }
      dyng::graph_properties p = props;
      p.num_weights = 1;
      graph_w g = graph_w::from_edges(res, list.view(), p);
      dyng::edge_batch<std::int32_t, std::int32_t> batch;
      for (std::uint64_t i = rng() % 7; i > 0; --i) {
        batch.delete_edge(vertex(rng), vertex(rng));
      }
      for (std::uint64_t i = rng() % 7; i > 0; --i) {
        batch.insert_edge(vertex(rng), vertex(rng), {static_cast<std::int32_t>(1 + rng() % 5)});
      }
      if (trial % 2 == 0 && batch.num_deletions() > 0) {
        batch.insert_edge(batch.delete_src()[0], batch.delete_dst()[0], {2});
      }
      SCOPED_TRACE(::testing::Message() << name << " trial " << trial);
      const edge_set before = edges_of(g);
      change_t change;
      dyng::detail::compute_structural_change(dyng::detail::graph_access::out_view(g), batch.view(),
                                              g.properties(), change);
      (void)g.apply(res, batch.view());
      const edge_set after = edges_of(g);
      // Sorted, without repeats; deletions exist before; (E_t minus D) plus I is E_{t+1}, and an
      // insertion is new unless it is also deleted (removed, then added back).
      EXPECT_TRUE(std::is_sorted(change.deletions.begin(), change.deletions.end()));
      EXPECT_TRUE(std::is_sorted(change.insertions.begin(), change.insertions.end()));
      EXPECT_EQ(std::adjacent_find(change.deletions.begin(), change.deletions.end()),
                change.deletions.end());
      EXPECT_EQ(std::adjacent_find(change.insertions.begin(), change.insertions.end()),
                change.insertions.end());
      edge_set model = before;
      edge_set deleted;
      for (const auto& d : change.deletions) {
        EXPECT_EQ(before.count({d.source, d.target}), 1U);
        model.erase({d.source, d.target});
        deleted.emplace(d.source, d.target);
      }
      for (const auto& c : change.insertions) {
        EXPECT_TRUE(model.count({c.source, c.target}) == 0U) << c.source << " " << c.target;
        EXPECT_TRUE(before.count({c.source, c.target}) == 0U ||
                    deleted.count({c.source, c.target}) == 1U);
        model.emplace(c.source, c.target);
      }
      EXPECT_EQ(model, after);
    }
  }
}

TEST(StructuralChange, RejectsUnsupportedGraphs) {
  const dyng::resources res = dyng::resources::sequential();
  dyng::edge_list<std::int32_t, unweighted> list;
  list.num_vertices = 2;
  list.add_edge(0, 1);
  const graph_u g =
      graph_u::from_edges(res, list.view(), dyng::graph_properties::mosp_compatible());
  const dyng::edge_batch<std::int32_t, unweighted> batch;
  change_t change;
  EXPECT_THROW(dyng::detail::compute_structural_change(dyng::detail::graph_access::out_view(g),
                                                       batch.view(), g.properties(), change),
               dyng::invalid_argument_error);
}

}  // namespace
