// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_path_costs_test.cpp
 * @brief The parallel path costs of the openmp and cuda backends (detail::mosp_path_costs_openmp)
 *        against the sequential traversal ported from mospPathCosts (detail::mosp_path_costs):
 *        the same costs on random trees of every shape (chains, stars, deep and wide levels,
 *        unreachable parts, parallel edges), and the same reported vertex when a tree edge is not
 *        an edge of the graph.
 */
#include "algorithms/mosp/problem.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/config.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using vertex_t = std::int32_t;
using edge_t = std::int32_t;
using weight_t = std::int32_t;
using workspace_t = dyng::detail::mosp_workspace<vertex_t, edge_t, weight_t>;

/// A graph that contains every tree edge (parent -> v), with extra edges and a parallel copy of
/// some tree edges (the first one in the row carries the cost), K weight columns.
struct tree_graph {
  std::vector<edge_t> row_ptr;
  std::vector<vertex_t> col_ind;
  std::vector<weight_t> weights;  // K columns
  std::vector<vertex_t> parent;
  int K = 0;

  [[nodiscard]] dyng::csr_view<vertex_t, edge_t, weight_t> view() const {
    dyng::csr_view<vertex_t, edge_t, weight_t> out;
    out.row_ptr = dyng::host_view(row_ptr);
    out.col_ind = dyng::host_view(col_ind);
    out.weights = dyng::host_view(weights);
    out.num_weights = K;
    return out;
  }
};

/// A random tree of `n` vertices rooted at the first vertex of a random permutation: the i-th gets a
/// parent among the `window` vertices before it (window 1: a chain; n: shallow and wide); a share
/// of `unreachable` vertices gets parent -1 (they and their subtrees are not reached).
tree_graph random_tree(vertex_t n, int K, vertex_t window, double unreachable, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<vertex_t> order(static_cast<std::size_t>(n));
  for (vertex_t v = 0; v < n; ++v) {
    order[static_cast<std::size_t>(v)] = v;
  }
  std::shuffle(order.begin(), order.end(), rng);
  tree_graph t;
  t.K = K;
  t.parent.assign(static_cast<std::size_t>(n), -1);
  std::vector<std::vector<vertex_t>> rows(static_cast<std::size_t>(n));
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  for (vertex_t i = 1; i < n; ++i) {
    const vertex_t lo = std::max<vertex_t>(0, i - window);
    std::uniform_int_distribution<vertex_t> pick(lo, i - 1);
    const vertex_t p = order[static_cast<std::size_t>(pick(rng))];
    const vertex_t v = order[static_cast<std::size_t>(i)];
    if (coin(rng) >= unreachable) {
      t.parent[static_cast<std::size_t>(v)] = p;
    }
    rows[static_cast<std::size_t>(p)].push_back(v);  // the edge exists either way
    if (coin(rng) < 0.05) {
      rows[static_cast<std::size_t>(p)].push_back(v);  // a parallel edge (not the first one)
    }
    if (coin(rng) < 0.3) {
      std::uniform_int_distribution<vertex_t> any(0, n - 1);
      rows[static_cast<std::size_t>(v)].push_back(any(rng));  // a non-tree edge
    }
  }
  t.row_ptr.push_back(0);
  for (const auto& row : rows) {
    t.col_ind.insert(t.col_ind.end(), row.begin(), row.end());
    t.row_ptr.push_back(static_cast<edge_t>(t.col_ind.size()));
  }
  std::uniform_int_distribution<weight_t> weight(1, 1000000);
  t.weights.resize(t.col_ind.size() * static_cast<std::size_t>(K));
  for (weight_t& w : t.weights) {
    w = weight(rng);
  }
  t.parent[static_cast<std::size_t>(order[0])] = -1;
  return t;
}

/// The costs and the return value of both versions on the same input.
struct both {
  std::vector<std::int64_t> sequential, parallel;
  vertex_t sequential_missing = -1, parallel_missing = -1;
};

both run_both(const dyng::resources& res, const tree_graph& t, vertex_t source) {
  const auto n = static_cast<std::size_t>(t.row_ptr.size() - 1);
  both out;
  out.sequential.assign(n * static_cast<std::size_t>(t.K), 7);
  out.parallel.assign(n * static_cast<std::size_t>(t.K), 7);
  workspace_t a;
  workspace_t b;
  a.reserve(n, t.K);
  b.reserve(n, t.K);
  out.sequential_missing = dyng::detail::mosp_path_costs(t.view(), t.parent.data(), source, t.K,
                                                         out.sequential.data(), a);
  out.parallel_missing = dyng::detail::mosp_path_costs_openmp(res, t.view(), t.parent.data(),
                                                              source, t.K, out.parallel.data(), b);
  return out;
}

class MospPathCosts : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    if (!DYNG_HAS_OPENMP) {
      GTEST_SKIP() << "the parallel path costs need the openmp backend";
    }
  }
};

INSTANTIATE_TEST_SUITE_P(Threads, MospPathCosts, ::testing::Values(2, 4, 13));

TEST_P(MospPathCosts, EqualTheSequentialTraversalOnRandomTrees) {
  const dyng::resources res = dyng::resources::openmp(GetParam());
  struct shape {
    vertex_t n;
    vertex_t window;
    double unreachable;
  };
  // Chains (one node per level), deep and narrow, shallow and wide, partly unreachable, and one
  // below the size from which threads are used.
  const std::vector<shape> shapes = {{40000, 1, 0.0},     {40000, 3, 0.0},   {60000, 50, 0.01},
                                     {60000, 60000, 0.0}, {50000, 400, 0.1}, {1000, 10, 0.0}};
  unsigned seed = 1;
  for (const shape& s : shapes) {
    for (int K : {1, 3}) {
      const tree_graph t = random_tree(s.n, K, s.window, s.unreachable, seed++);
      // Root the traversal at the parentless vertex with the most children (the permutation's
      // first vertex, unless an unreachable vertex has more).
      vertex_t root = 0;
      std::vector<std::int64_t> children(t.parent.size(), 0);
      for (vertex_t p : t.parent) {
        if (p >= 0) {
          ++children[static_cast<std::size_t>(p)];
        }
      }
      for (std::size_t v = 0; v < t.parent.size(); ++v) {
        if (t.parent[v] < 0 && (t.parent[static_cast<std::size_t>(root)] >= 0 ||
                                children[v] > children[static_cast<std::size_t>(root)])) {
          root = static_cast<vertex_t>(v);
        }
      }
      const both r = run_both(res, t, root);
      EXPECT_EQ(r.sequential_missing, -1);
      EXPECT_EQ(r.parallel_missing, -1);
      EXPECT_TRUE(r.parallel == r.sequential)
          << "n " << s.n << " window " << s.window << " K " << K;
    }
  }
}

TEST_P(MospPathCosts, AMissingTreeEdgeIsReportedAsTheSequentialTraversalReportsIt) {
  const dyng::resources res = dyng::resources::openmp(GetParam());
  for (unsigned seed = 100; seed < 106; ++seed) {
    tree_graph t = random_tree(50000, 2, seed % 2 == 0 ? 5 : 2000, 0.0, seed);
    vertex_t root = 0;
    for (std::size_t v = 0; v < t.parent.size(); ++v) {
      if (t.parent[v] < 0) {
        root = static_cast<vertex_t>(v);
      }
    }
    // Point several vertices at parents they have no edge from.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<vertex_t> any(0, 49999);
    for (int i = 0; i < 3; ++i) {
      const vertex_t v = any(rng);
      const vertex_t p = t.parent[static_cast<std::size_t>(v)];
      if (v == root || p < 0) {
        continue;
      }
      const auto row_begin = t.col_ind.begin() + t.row_ptr[static_cast<std::size_t>(p)];
      const auto row_end = t.col_ind.begin() + t.row_ptr[static_cast<std::size_t>(p) + 1];
      std::replace(row_begin, row_end, v, root);  // the edge p -> v becomes p -> root
    }
    const both r = run_both(res, t, root);
    EXPECT_EQ(r.parallel_missing, r.sequential_missing) << "seed " << seed;
  }
}

}  // namespace
