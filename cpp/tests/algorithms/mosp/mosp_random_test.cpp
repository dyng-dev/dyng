// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_random_test.cpp
 * @brief Randomized differential tests of mosp: after compute() and after every batch of a chain,
 *        the result equals the independent references (dyng::testing: Dijkstra per objective,
 *        combined_graph_reference() of the K trees and Dijkstra on it, mosp_path_costs_reference())
 *        and a fresh compute() on the new graph, and every backend agrees with the sequential one
 *        bit for bit (trees, MOSP tree, path costs, the deterministic counters). In the CUDA test
 *        executable the cuda backend joins twice: with the fused and with the operators sssp
 *        engine (conformance check C4).
 *
 * K in {1, 2, 3, 4}, preferences none or random (lcm <= 2^20), graph shapes sparse / dense / grid
 * / chain, MOSP's graphs (mosp_compatible(): parallel edges, self-loops, insertion order) and
 * simple sorted graphs under upsert and set() semantics, weight ranges with many ties, batch mixes
 * of insertions, deletions (random and on tree edges), re-weightings, delete-all and vertex growth,
 * 3 to 5 batches per chain. A second suite starts from trees with non-lowest tie parents
 * (from_arrays with canonicalize = false, as MOSP's dataset trees): the backends agree, and the
 * MOSP tree is the canonical tree of the combined graph of the K updated trees.
 *
 * Every failure prints its seed; replay one with DYNG_TEST_SEED=<seed>.
 */
#include "support/gtest_helpers.hpp"
#include "support/test_seeds.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/mosp.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/testing/mosp_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::test::seed_trace;
using dyng::test::test_seeds;

/// A host copy of everything a mosp result holds.
struct snapshot {
  std::vector<std::vector<std::int64_t>> distances;
  std::vector<std::vector<std::int64_t>> parents;
  std::vector<std::int64_t> combined_distances;
  std::vector<std::int64_t> combined_parents;
  std::vector<std::int64_t> costs;

  bool operator==(const snapshot& o) const {
    return distances == o.distances && parents == o.parents &&
           combined_distances == o.combined_distances && combined_parents == o.combined_parents &&
           costs == o.costs;
  }
};

template <typename vertex_t>
snapshot take(const dyng::mosp::result<vertex_t>& r) {
  snapshot s;
  for (int k = 0; k < r.num_objectives(); ++k) {
    s.distances.push_back(dyng::test::host_copy(r.distances(k)));
    const auto p = dyng::test::host_copy(r.parents(k));
    s.parents.emplace_back(p.begin(), p.end());
  }
  s.combined_distances = dyng::test::host_copy(r.combined_distances());
  const auto cp = dyng::test::host_copy(r.combined_parents());
  s.combined_parents.assign(cp.begin(), cp.end());
  s.costs = dyng::test::host_copy(r.path_costs());
  return s;
}

/// The references of a graph: Dijkstra per objective (or the given trees), the combined graph of
/// the trees and its Dijkstra tree, the path costs along it.
template <typename vertex_t, typename edge_t, typename weight_t>
snapshot reference(const dyng::csr<vertex_t, edge_t, weight_t>& g, vertex_t source, int K,
                   const std::vector<std::int32_t>& preferences,
                   const std::vector<std::vector<vertex_t>>* trees = nullptr) {
  snapshot s;
  std::vector<std::vector<vertex_t>> parents;
  for (int k = 0; k < K; ++k) {
    if (trees == nullptr) {
      const auto tree = dyng::testing::dijkstra(g.view(), source, k);
      s.distances.push_back(tree.distances);
      parents.push_back(tree.parents);
    } else {
      parents.push_back((*trees)[static_cast<std::size_t>(k)]);
    }
    s.parents.emplace_back(parents.back().begin(), parents.back().end());
  }
  const auto combined = dyng::testing::combined_graph_reference(parents, source, preferences);
  const auto mosp_tree = dyng::testing::dijkstra(combined.view(), source);
  s.combined_distances = mosp_tree.distances;
  s.combined_parents.assign(mosp_tree.parents.begin(), mosp_tree.parents.end());
  s.costs = dyng::testing::mosp_path_costs_reference(g.view(), mosp_tree.parents, source, K);
  return s;
}

/// The deterministic counters of an update.
std::vector<std::int64_t> counters(const dyng::mosp::stats& s) {
  std::vector<std::int64_t> out{s.affected,
                                s.combined_edges,
                                s.preference_scale,
                                s.batch.inserted_edges,
                                s.batch.deleted_edges,
                                s.batch.updated_edges};
  for (const auto& o : s.objectives) {
    out.push_back(o.invalidated);
    out.push_back(o.affected);
  }
  return out;
}

enum class shape { sparse, dense, grid, chain };

template <typename vertex_t, typename weight_t>
struct generator {
  std::mt19937_64 rng;
  int num_weights = 1;
  weight_t max_weight = 9;

  std::int64_t uniform(std::int64_t lo, std::int64_t hi) {
    return std::uniform_int_distribution<std::int64_t>(lo, hi)(rng);
  }
  bool coin(double p) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p;
  }
  std::vector<weight_t> weights() {
    std::vector<weight_t> w;
    for (int k = 0; k < num_weights; ++k) {
      w.push_back(static_cast<weight_t>(uniform(1, max_weight)));
    }
    return w;
  }
  void push(dyng::edge_list<vertex_t, weight_t>& edges, std::int64_t u, std::int64_t v) {
    edges.src.push_back(static_cast<vertex_t>(u));
    edges.dst.push_back(static_cast<vertex_t>(v));
    for (const weight_t w : weights()) {
      edges.weights.push_back(w);
    }
  }

  dyng::edge_list<vertex_t, weight_t> graph_edges(shape s, bool simple) {
    dyng::edge_list<vertex_t, weight_t> edges;
    edges.num_weights = num_weights;
    std::int64_t n = 1;
    switch (s) {
      case shape::sparse:
        n = uniform(1, 60);
        for (std::int64_t i = uniform(0, 3 * n); i > 0; --i) {
          push(edges, uniform(0, n - 1), uniform(0, n - 1));
        }
        break;
      case shape::dense:
        n = uniform(2, 16);
        for (std::int64_t u = 0; u < n; ++u) {
          for (std::int64_t v = 0; v < n; ++v) {
            if (coin(0.45)) {
              push(edges, u, v);
            }
          }
        }
        break;
      case shape::grid: {
        const std::int64_t w = uniform(1, 8);
        const std::int64_t h = uniform(1, 8);
        n = w * h;
        for (std::int64_t y = 0; y < h; ++y) {
          for (std::int64_t x = 0; x < w; ++x) {
            if (x + 1 < w) {
              push(edges, y * w + x, y * w + x + 1);
              push(edges, y * w + x + 1, y * w + x);
            }
            if (y + 1 < h) {
              push(edges, y * w + x, (y + 1) * w + x);
              push(edges, (y + 1) * w + x, y * w + x);
            }
          }
        }
        break;
      }
      case shape::chain:
        n = uniform(2, 60);
        for (std::int64_t v = 0; v + 1 < n; ++v) {
          push(edges, v, v + 1);
          if (coin(0.25)) {
            push(edges, v + 1, uniform(0, v));
          }
          if (coin(0.15)) {
            push(edges, v, uniform(v, n - 1));
          }
        }
        break;
    }
    edges.num_vertices = static_cast<vertex_t>(n);
    if (simple) {  // sorted rows without parallel edges or self-loops: the first copy of a pair
      dyng::edge_list<vertex_t, weight_t> kept;
      kept.num_weights = num_weights;
      kept.num_vertices = edges.num_vertices;
      std::vector<std::pair<vertex_t, vertex_t>> seen;
      for (std::size_t i = 0; i < edges.src.size(); ++i) {
        const auto pair = std::make_pair(edges.src[i], edges.dst[i]);
        if (pair.first == pair.second || std::find(seen.begin(), seen.end(), pair) != seen.end()) {
          continue;
        }
        seen.push_back(pair);
        kept.src.push_back(pair.first);
        kept.dst.push_back(pair.second);
        for (int k = 0; k < num_weights; ++k) {
          kept.weights.push_back(edges.weights[i * static_cast<std::size_t>(num_weights) +
                                               static_cast<std::size_t>(k)]);
        }
      }
      return kept;
    }
    return edges;
  }

  /// A random batch for the current graph `g`; `tree` is one objective's tree.
  template <typename edge_t>
  dyng::edge_batch<vertex_t, weight_t> batch(const dyng::csr<vertex_t, edge_t, weight_t>& g,
                                             const std::vector<std::int64_t>& tree, bool simple) {
    dyng::edge_batch<vertex_t, weight_t> b(num_weights);
    const std::int64_t n = g.num_vertices();
    const std::int64_t m = g.num_edges();
    std::vector<vertex_t> src(static_cast<std::size_t>(m));
    for (std::int64_t u = 0; u < n; ++u) {
      for (edge_t e = g.row_ptr[static_cast<std::size_t>(u)];
           e < g.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
        src[static_cast<std::size_t>(e)] = static_cast<vertex_t>(u);
      }
    }
    const auto insert = [&](std::int64_t u, std::int64_t v) {
      if (u == v && simple) {
        return;
      }
      const std::vector<weight_t> w = weights();
      b.insert_edge(static_cast<vertex_t>(u), static_cast<vertex_t>(v), dyng::host_view(w));
    };
    const std::int64_t count = uniform(0, std::max<std::int64_t>(2, (n + m) / 3));
    switch (uniform(0, 6)) {
      case 0:
        for (std::int64_t i = 0; i < count && n > 0; ++i) {
          insert(uniform(0, n - 1), uniform(0, n - 1));
        }
        break;
      case 1:
        for (std::int64_t i = 0; i < count && m > 0; ++i) {
          const auto e = static_cast<std::size_t>(uniform(0, m - 1));
          b.delete_edge(src[e], g.col_ind[e]);
        }
        break;
      case 2:
        for (std::int64_t v = 0; v < n; ++v) {
          if (tree[static_cast<std::size_t>(v)] >= 0 && coin(0.3)) {
            b.delete_edge(static_cast<vertex_t>(tree[static_cast<std::size_t>(v)]),
                          static_cast<vertex_t>(v));
          }
        }
        break;
      case 3:  // re-weighting (upsert semantics; ignored as insertions of existing edges by set())
        for (std::int64_t i = 0; i < count && m > 0; ++i) {
          const auto e = static_cast<std::size_t>(uniform(0, m - 1));
          insert(src[e], g.col_ind[e]);
        }
        break;
      case 4:
        for (std::int64_t e = 0; e < m; ++e) {
          b.delete_edge(src[static_cast<std::size_t>(e)], g.col_ind[static_cast<std::size_t>(e)]);
        }
        break;
      default:
        for (std::int64_t i = 0; i < count; ++i) {
          if (m > 0 && coin(0.5)) {
            const auto e = static_cast<std::size_t>(uniform(0, m - 1));
            b.delete_edge(src[e], g.col_ind[e]);
          } else if (n > 0) {
            insert(uniform(0, n - 1), uniform(0, n - 1));
          }
        }
        break;
    }
    if (coin(0.2) && n > 0) {  // vertex growth
      insert(uniform(0, n - 1), n);
      if (coin(0.5)) {
        insert(n, uniform(0, n - 1));
      }
    }
    return b;
  }

  /// Random preferences for K objectives (none, or values whose lcm is at most 2^20).
  std::vector<std::int32_t> preferences(int K) {
    if (coin(0.3)) {
      return {};
    }
    std::vector<std::int32_t> p;
    std::int64_t scale = 1;
    for (int k = 0; k < K; ++k) {
      std::int32_t x = 1;
      do {
        x = static_cast<std::int32_t>(uniform(1, coin(0.2) ? 64 : 6));
      } while (std::lcm(scale, static_cast<std::int64_t>(x)) > dyng::mosp::max_preference_scale);
      scale = std::lcm(scale, static_cast<std::int64_t>(x));
      p.push_back(x);
    }
    return p;
  }
};

/// One backend (and CUDA engine) under test.
struct config {
  dyng::resources res;
  dyng::engine engine = dyng::engine::automatic;
  std::string label;
};

std::vector<config> configs() {
  std::vector<config> out;
  for (const dyng::backend b : dyng::test::comparison_backends()) {
    const dyng::resources res = dyng::test::make_resources(b, b == dyng::backend::openmp ? 4 : 0);
    if (b == dyng::backend::cuda) {
      out.push_back({res, dyng::engine::fused, "cuda-fused"});
      out.push_back({res, dyng::engine::operators, "cuda-operators"});
    } else {
      out.push_back({res, dyng::engine::automatic, std::string(dyng::to_string(b))});
    }
  }
  return out;
}

template <typename vertex_t, typename edge_t>
void run_chains(std::uint64_t base, int count) {
  using weight_t = std::int32_t;
  using graph_t = dyng::graph<vertex_t, edge_t, weight_t>;
  const std::vector<config> all = configs();
  for (const std::uint64_t seed : test_seeds(base, count)) {
    SCOPED_TRACE(seed_trace(seed));
    generator<vertex_t, weight_t> gen{std::mt19937_64(seed)};
    const int K = static_cast<int>(gen.uniform(1, 4));
    gen.num_weights = K;
    gen.max_weight = static_cast<weight_t>(gen.coin(0.5) ? 4 : 30);
    const auto s = static_cast<shape>(gen.uniform(0, 3));
    const int props_kind = static_cast<int>(gen.uniform(0, 2));
    dyng::graph_properties props =
        props_kind == 0 ? dyng::graph_properties::mosp_compatible() : dyng::graph_properties{};
    if (props_kind == 2) {
      props.semantics = dyng::batch_semantics::set();
    }
    props.store_transposed = true;
    const bool simple = props_kind != 0;
    const auto edges = gen.graph_edges(s, simple);
    dyng::mosp::options opt;
    opt.preferences = gen.preferences(K);
    SCOPED_TRACE("K " + std::to_string(K) + ", shape " + std::to_string(static_cast<int>(s)) +
                 ", properties " + std::to_string(props_kind) + ", preferences " +
                 std::to_string(opt.preferences.size()));
    std::vector<graph_t> graphs;
    std::vector<dyng::mosp::result<vertex_t>> results;
    for (const config& c : all) {
      graphs.push_back(graph_t::from_edges(c.res, edges.view(), props));
      dyng::mosp::options o = opt;
      o.cuda_engine = c.engine;
      results.push_back(dyng::mosp::compute(c.res, graphs.back(), vertex_t{0}, o));
    }
    const auto host = graphs.front().to_csr(all.front().res);
    const snapshot expected = reference(host, vertex_t{0}, K, opt.preferences);
    for (std::size_t i = 0; i < all.size(); ++i) {
      ASSERT_TRUE(take(results[i]) == expected) << "compute() on " << all[i].label;
    }
    const int steps = static_cast<int>(gen.uniform(3, 5));
    for (int step = 0; step < steps; ++step) {
      SCOPED_TRACE("batch " + std::to_string(step));
      const auto current = graphs.front().to_csr(all.front().res);
      const auto batch = gen.batch(current, take(results.front()).parents.front(), simple);
      std::vector<std::int64_t> reference_counters;
      for (std::size_t i = 0; i < all.size(); ++i) {
        SCOPED_TRACE(all[i].label);
        const dyng::mosp::stats st =
            dyng::mosp::update(all[i].res, graphs[i], batch.view(), results[i]);
        if (i == 0) {
          reference_counters = counters(st);
        } else {
          EXPECT_EQ(counters(st), reference_counters);
        }
        EXPECT_EQ(static_cast<int>(st.objectives.size()), K);
      }
      const auto updated = graphs.front().to_csr(all.front().res);
      const snapshot want = reference(updated, vertex_t{0}, K, opt.preferences);
      for (std::size_t i = 0; i < all.size(); ++i) {
        ASSERT_TRUE(take(results[i]) == want) << "update chain on " << all[i].label;
      }
      dyng::mosp::options o = opt;
      const auto fresh = dyng::mosp::compute(all.front().res, graphs.front(), vertex_t{0}, o);
      ASSERT_TRUE(take(fresh) == want) << "compute() of the new graph";
    }
  }
}

TEST(MospRandom, ChainsEqualTheReferencesOnEveryBackend) {
  run_chains<std::int32_t, std::int32_t>(70000, 60);
}

TEST(MospRandom, ChainsEqualTheReferencesWith64BitOffsets) {
  run_chains<std::int32_t, std::int64_t>(71000, 12);
}

TEST(MospRandom, ChainsEqualTheReferencesWith64BitIds) {
  run_chains<std::int64_t, std::int64_t>(72000, 12);
}

/// Trees with non-lowest tie parents (from_arrays, canonicalize = false): the backends agree, and
/// the MOSP tree is the canonical tree of the combined graph of the K updated trees.
TEST(MospRandom, NonCanonicalInputTreesAgreeAcrossBackends) {
  using vertex_t = std::int32_t;
  using weight_t = std::int32_t;
  using graph_t = dyng::graph<vertex_t, std::int32_t, weight_t>;
  const std::vector<config> all = configs();
  for (const std::uint64_t seed : test_seeds(73000, 30)) {
    SCOPED_TRACE(seed_trace(seed));
    generator<vertex_t, weight_t> gen{std::mt19937_64(seed)};
    const int K = static_cast<int>(gen.uniform(1, 4));
    gen.num_weights = K;
    gen.max_weight = 3;  // many ties
    const auto edges = gen.graph_edges(static_cast<shape>(gen.uniform(0, 3)), false);
    dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
    dyng::mosp::options opt;
    opt.preferences = gen.preferences(K);
    // Valid trees with random tight parents (computed on the host, imported everywhere).
    const graph_t g0 = graph_t::from_edges(dyng::resources::sequential(), edges.view(), props);
    const auto csr0 = g0.to_csr(dyng::resources::sequential());
    std::vector<std::vector<std::int64_t>> dist(static_cast<std::size_t>(K));
    std::vector<std::vector<vertex_t>> par(static_cast<std::size_t>(K));
    const auto m = static_cast<std::size_t>(csr0.num_edges());
    for (int k = 0; k < K; ++k) {
      const auto tree = dyng::testing::dijkstra(csr0.view(), vertex_t{0}, k);
      dist[static_cast<std::size_t>(k)] = tree.distances;
      std::vector<vertex_t> p = tree.parents;
      for (vertex_t u = 0; u < csr0.num_vertices(); ++u) {
        if (tree.distances[static_cast<std::size_t>(u)] >=
            dyng::infinite_distance<std::int64_t>() / 2) {
          continue;
        }
        for (auto e = csr0.row_ptr[static_cast<std::size_t>(u)];
             e < csr0.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
          const vertex_t v = csr0.col_ind[static_cast<std::size_t>(e)];
          const auto w =
              csr0.weights[static_cast<std::size_t>(k) * m + static_cast<std::size_t>(e)];
          if (v != 0 &&
              tree.distances[static_cast<std::size_t>(u)] + w ==
                  tree.distances[static_cast<std::size_t>(v)] &&
              gen.coin(0.5)) {
            p[static_cast<std::size_t>(v)] = u;
          }
        }
      }
      par[static_cast<std::size_t>(k)] = std::move(p);
    }
    std::vector<dyng::array_view<const std::int64_t>> dv;
    std::vector<dyng::array_view<const vertex_t>> pv;
    for (int k = 0; k < K; ++k) {
      dv.push_back(dyng::host_view(dist[static_cast<std::size_t>(k)]));
      pv.push_back(dyng::host_view(par[static_cast<std::size_t>(k)]));
    }
    std::vector<graph_t> graphs;
    std::vector<dyng::mosp::result<vertex_t>> results;
    for (const config& c : all) {
      graphs.push_back(graph_t::from_edges(c.res, edges.view(), props));
      dyng::mosp::options o = opt;
      o.cuda_engine = c.engine;
      results.push_back(dyng::mosp::result<vertex_t>::from_arrays(
          c.res, graphs.back(), vertex_t{0}, dyng::host_view(std::as_const(dv)),
          dyng::host_view(std::as_const(pv)), false, o));
    }
    for (int step = 0; step < 3; ++step) {
      SCOPED_TRACE("batch " + std::to_string(step));
      const auto current = graphs.front().to_csr(all.front().res);
      const auto batch = gen.batch(current, take(results.front()).parents.front(), false);
      for (std::size_t i = 0; i < all.size(); ++i) {
        (void)dyng::mosp::update(all[i].res, graphs[i], batch.view(), results[i]);
      }
      const snapshot first = take(results.front());
      for (std::size_t i = 1; i < all.size(); ++i) {
        ASSERT_TRUE(take(results[i]) == first) << all[i].label << " differs from sequential";
      }
      // The MOSP tree and the costs from the K (non-canonical) trees.
      std::vector<std::vector<vertex_t>> trees;
      for (const auto& p : first.parents) {
        trees.emplace_back(p.begin(), p.end());
      }
      const auto updated = graphs.front().to_csr(all.front().res);
      const snapshot want = reference(updated, vertex_t{0}, K, opt.preferences, &trees);
      EXPECT_EQ(first.combined_distances, want.combined_distances);
      EXPECT_EQ(first.combined_parents, want.combined_parents);
      EXPECT_EQ(first.costs, want.costs);
    }
  }
}

}  // namespace
