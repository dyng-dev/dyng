// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_random_test.cpp
 * @brief Randomized differential tests of sssp: every update chain equals Dijkstra with lowest-id
 *        ties (testing::check_sssp_tree, canonical mode) and sssp::compute() on the new graph, and
 *        the sequential and OpenMP backends agree bit for bit (distances, parents and the
 *        deterministic counters), also from valid input trees with non-lowest tie parents
 *        (from_arrays with canonicalize = false).
 *
 * Graph shapes: sparse and dense random graphs, road-like grids, hub-heavy graphs and long chains,
 * with parallel edges and self-loops under graph_properties::mosp_compatible(), simple sorted
 * graphs under the default properties, and undirected graphs. Weight ranges: tiny (many ties),
 * [1, 100] and up to 2^31 - 1 (the distance-only packing fallback). Batch mixes: insertions,
 * deletions (random and on tree edges), mixed, re-weighting (increases and decreases on tree
 * edges), delete-all, missing deletions, vertex growth; 3 to 5 consecutive batches.
 *
 * Every failure prints its seed; replay one with DYNG_TEST_SEED=<seed> (and DYNG_TEST_SEEDS=<n>
 * for the number of seeds per test).
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>
#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

/// Seeds of a randomized test: DYNG_TEST_SEED replays one seed; DYNG_TEST_SEEDS sets the count.
std::vector<std::uint64_t> seeds(std::uint64_t base, int default_count) {
  if (const char* one = std::getenv("DYNG_TEST_SEED")) {
    return {std::strtoull(one, nullptr, 10)};
  }
  int count = default_count;
  if (const char* n = std::getenv("DYNG_TEST_SEEDS")) {
    count = std::max(1, std::atoi(n));
  }
  std::vector<std::uint64_t> out;
  for (int i = 0; i < count; ++i) {
    out.push_back(base + static_cast<std::uint64_t>(i));
  }
  return out;
}

enum class shape { sparse, dense, grid, hub, chain };
constexpr shape all_shapes[] = {shape::sparse, shape::dense, shape::grid, shape::hub, shape::chain};

const char* shape_name(shape s) {
  switch (s) {
    case shape::sparse:
      return "sparse";
    case shape::dense:
      return "dense";
    case shape::grid:
      return "grid";
    case shape::hub:
      return "hub";
    case shape::chain:
      return "chain";
  }
  return "?";
}

template <typename vertex_t, typename weight_t>
struct generator {
  std::mt19937_64 rng;
  int num_weights = 1;
  weight_t max_weight = 100;

  std::int64_t uniform(std::int64_t lo, std::int64_t hi) {
    return std::uniform_int_distribution<std::int64_t>(lo, hi)(rng);
  }
  bool coin(double p) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p;
  }
  weight_t weight() {
    return static_cast<weight_t>(uniform(1, max_weight));
  }
  void push_edge(dyng::edge_list<vertex_t, weight_t>& edges, std::int64_t u, std::int64_t v) {
    edges.src.push_back(static_cast<vertex_t>(u));
    edges.dst.push_back(static_cast<vertex_t>(v));
    for (int k = 0; k < num_weights; ++k) {
      edges.weights.push_back(weight());
    }
  }

  dyng::edge_list<vertex_t, weight_t> graph_edges(shape s, bool self_loops) {
    dyng::edge_list<vertex_t, weight_t> edges;
    edges.num_weights = num_weights;
    std::int64_t n = 1;
    switch (s) {
      case shape::sparse: {
        n = uniform(1, 60);
        const std::int64_t m = uniform(0, 3 * n);
        for (std::int64_t i = 0; i < m; ++i) {
          push_edge(edges, uniform(0, n - 1), uniform(0, n - 1));
        }
        break;
      }
      case shape::dense: {
        n = uniform(2, 18);
        for (std::int64_t u = 0; u < n; ++u) {
          for (std::int64_t v = 0; v < n; ++v) {
            if (coin(0.45)) {
              push_edge(edges, u, v);
            }
          }
        }
        break;
      }
      case shape::grid: {
        const std::int64_t w = uniform(1, 9);
        const std::int64_t h = uniform(1, 9);
        n = w * h;
        for (std::int64_t y = 0; y < h; ++y) {
          for (std::int64_t x = 0; x < w; ++x) {
            if (x + 1 < w && !coin(0.1)) {
              push_edge(edges, y * w + x, y * w + x + 1);
              push_edge(edges, y * w + x + 1, y * w + x);
            }
            if (y + 1 < h && !coin(0.1)) {
              push_edge(edges, y * w + x, (y + 1) * w + x);
              push_edge(edges, (y + 1) * w + x, y * w + x);
            }
          }
        }
        break;
      }
      case shape::hub: {
        n = uniform(3, 50);
        const std::int64_t hubs = uniform(1, 3);
        for (std::int64_t v = 0; v < n; ++v) {
          for (std::int64_t h = 0; h < hubs; ++h) {
            if (coin(0.7)) {
              push_edge(edges, h, v);
            }
            if (coin(0.3)) {
              push_edge(edges, v, h);
            }
          }
          if (coin(0.3)) {
            push_edge(edges, v, uniform(0, n - 1));
          }
        }
        break;
      }
      case shape::chain: {
        n = uniform(2, 70);
        for (std::int64_t v = 0; v + 1 < n; ++v) {
          push_edge(edges, v, v + 1);
          if (coin(0.25)) {
            push_edge(edges, v + 1, uniform(0, v));  // back edges
          }
          if (coin(0.15)) {
            push_edge(edges, v, uniform(v, n - 1));  // shortcuts
          }
        }
        break;
      }
    }
    if (!self_loops) {
      dyng::edge_list<vertex_t, weight_t> kept;
      kept.num_weights = num_weights;
      for (std::size_t i = 0; i < edges.src.size(); ++i) {
        if (edges.src[i] != edges.dst[i]) {
          kept.src.push_back(edges.src[i]);
          kept.dst.push_back(edges.dst[i]);
          for (int k = 0; k < num_weights; ++k) {
            kept.weights.push_back(edges.weights[i * static_cast<std::size_t>(num_weights) +
                                                 static_cast<std::size_t>(k)]);
          }
        }
      }
      edges = std::move(kept);
    }
    edges.num_vertices = static_cast<vertex_t>(n);
    return edges;
  }

  /// A random batch for the current graph; `tree` is one tree (targets for deletions).
  template <typename edge_t>
  dyng::edge_batch<vertex_t, weight_t> batch(const dyng::csr<vertex_t, edge_t, weight_t>& g,
                                             const std::vector<vertex_t>& tree, bool growth,
                                             bool self_loops) {
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
    auto new_weights = [&] {
      std::vector<weight_t> w;
      for (int k = 0; k < num_weights; ++k) {
        w.push_back(weight());
      }
      return w;
    };
    auto insert = [&](std::int64_t u, std::int64_t v) {
      if (u == v && !self_loops) {
        return;
      }
      const std::vector<weight_t> w = new_weights();
      b.insert_edge(static_cast<vertex_t>(u), static_cast<vertex_t>(v), dyng::host_view(w));
    };
    const int mix = static_cast<int>(uniform(0, 7));
    const std::int64_t count = uniform(0, std::max<std::int64_t>(2, (n + m) / 3));
    switch (mix) {
      case 0:  // insertions only
        for (std::int64_t i = 0; i < count && n > 0; ++i) {
          insert(uniform(0, n - 1), uniform(0, n - 1));
        }
        break;
      case 1:  // random deletions (some of them missing)
        for (std::int64_t i = 0; i < count && m > 0; ++i) {
          const auto e = static_cast<std::size_t>(uniform(0, m - 1));
          b.delete_edge(src[e], g.col_ind[e]);
          if (coin(0.1) && n > 0) {
            b.delete_edge(static_cast<vertex_t>(uniform(0, n - 1)),
                          static_cast<vertex_t>(uniform(0, n - 1)));
          }
        }
        break;
      case 2:  // tree-edge deletions
        for (std::int64_t v = 0; v < n; ++v) {
          if (tree[static_cast<std::size_t>(v)] >= 0 && coin(0.3)) {
            b.delete_edge(tree[static_cast<std::size_t>(v)], static_cast<vertex_t>(v));
          }
        }
        break;
      case 3:  // re-weighting of existing edges (increases and decreases)
        for (std::int64_t i = 0; i < count && m > 0; ++i) {
          const auto e = static_cast<std::size_t>(uniform(0, m - 1));
          insert(src[e], g.col_ind[e]);
        }
        break;
      case 4:  // tree-edge weight increases (and a few decreases)
        for (std::int64_t v = 0; v < n; ++v) {
          const vertex_t p = tree[static_cast<std::size_t>(v)];
          if (p >= 0 && coin(0.35)) {
            std::vector<weight_t> w = new_weights();
            for (auto& x : w) {
              x = static_cast<weight_t>(std::min<std::int64_t>(max_weight, x + uniform(0, 50)));
            }
            b.insert_edge(p, static_cast<vertex_t>(v), dyng::host_view(w));
          }
        }
        break;
      case 5:  // delete everything
        for (std::int64_t e = 0; e < m; ++e) {
          b.delete_edge(src[static_cast<std::size_t>(e)], g.col_ind[static_cast<std::size_t>(e)]);
        }
        break;
      default:  // mixed, deletions first then insertions (possibly re-inserting deleted edges)
        for (std::int64_t i = 0; i < count; ++i) {
          if (m > 0 && coin(0.5)) {
            const auto e = static_cast<std::size_t>(uniform(0, m - 1));
            b.delete_edge(src[e], g.col_ind[e]);
            if (coin(0.3)) {
              insert(src[e], g.col_ind[e]);
            }
          } else if (n > 0) {
            insert(uniform(0, n - 1), uniform(0, n - 1));
          }
        }
        break;
    }
    if (growth && coin(0.3) && n > 0) {
      const std::int64_t extra = uniform(1, 3);
      for (std::int64_t i = 0; i < extra; ++i) {
        insert(uniform(0, n - 1), n + i);
        if (coin(0.5)) {
          insert(n + i, uniform(0, n - 1));
        }
      }
    }
    return b;
  }
};

struct scenario {
  bool mosp_compatible = true;
  bool directed = true;
  bool growth = false;
  /// Start from a valid shortest-path tree whose tie parents are random tight in-neighbours
  /// (not the lowest ids), adopted with from_arrays(canonicalize = false), as MOSP's datasets
  /// are. The backends must still agree bit for bit (they share sospUpdateCpu's tie rule), the
  /// distances must equal compute(), and the tree must stay a valid shortest-path tree.
  bool perturb_ties = false;
};

/// A valid shortest-path tree of `g` (objective k) with random tight parents instead of the
/// lowest ids. Weights are >= 1, so every tight parent is strictly closer: the tree is acyclic.
template <typename vertex_t, typename edge_t, typename weight_t, typename rng_t>
std::vector<vertex_t> perturb_tie_parents(const dyng::csr<vertex_t, edge_t, weight_t>& g, int k,
                                          vertex_t source,
                                          dyng::array_view<const std::int64_t> distances,
                                          dyng::array_view<const vertex_t> parents, rng_t& rng) {
  const std::int64_t n = g.num_vertices();
  const auto m = static_cast<std::size_t>(g.num_edges());
  std::vector<std::vector<vertex_t>> tight(static_cast<std::size_t>(n));
  constexpr std::int64_t unreachable = dyng::infinite_distance<std::int64_t>() / 2;
  for (std::int64_t u = 0; u < n; ++u) {
    const std::int64_t du = distances[static_cast<std::size_t>(u)];
    if (du >= unreachable) {
      continue;
    }
    for (edge_t e = g.row_ptr[static_cast<std::size_t>(u)];
         e < g.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
      const vertex_t v = g.col_ind[static_cast<std::size_t>(e)];
      const auto w = static_cast<std::int64_t>(
          g.weights[static_cast<std::size_t>(k) * m + static_cast<std::size_t>(e)]);
      if (v != source && du + w == distances[static_cast<std::size_t>(v)]) {
        tight[static_cast<std::size_t>(v)].push_back(static_cast<vertex_t>(u));
      }
    }
  }
  std::vector<vertex_t> out(parents.begin(), parents.end());
  for (std::int64_t v = 0; v < n; ++v) {
    const auto& t = tight[static_cast<std::size_t>(v)];
    if (!t.empty()) {
      out[static_cast<std::size_t>(v)] =
          t[std::uniform_int_distribution<std::size_t>(0, t.size() - 1)(rng)];
    }
  }
  return out;
}

template <typename graph_t>
class SsspRandom : public ::testing::Test {};

using graph_types = ::testing::Types<dyng::graph<std::int32_t, std::int32_t, std::int32_t>,
                                     dyng::graph<std::int32_t, std::int64_t, std::int32_t>,
                                     dyng::graph<std::int64_t, std::int64_t, std::int32_t>>;
TYPED_TEST_SUITE(SsspRandom, graph_types, dyng::test::type_index_name);

template <typename graph_t>
void run_scenario(std::uint64_t seed, shape s, const scenario& sc) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;
  SCOPED_TRACE("seed " + std::to_string(seed) + " (replay: DYNG_TEST_SEED=" + std::to_string(seed) +
               "), shape " + shape_name(s) +
               (sc.mosp_compatible ? ", mosp_compatible" : ", simple sorted") +
               (sc.directed ? ", directed" : ", undirected") + (sc.growth ? ", growth" : ""));
  generator<vertex_t, weight_t> gen{std::mt19937_64(seed)};
  gen.num_weights = static_cast<int>(gen.uniform(1, 3));
  const int weight_class = static_cast<int>(gen.uniform(0, 9));
  gen.max_weight = weight_class < 4   ? static_cast<weight_t>(gen.uniform(1, 3))
                   : weight_class < 9 ? weight_t{100}
                                      : weight_t{2147483647};
  const auto edges = gen.graph_edges(s, sc.mosp_compatible);
  dyng::graph_properties props =
      sc.mosp_compatible ? dyng::graph_properties::mosp_compatible() : dyng::graph_properties{};
  props.directed = sc.directed;
  props.semantics.allow_vertex_growth = sc.growth;

  const std::vector<dyng::backend> backends = dyng::test::host_backends();
  const int threads = static_cast<int>(gen.uniform(1, 8));
  std::vector<dyng::resources> res;
  std::vector<graph_t> graphs;
  for (dyng::backend b : backends) {
    res.push_back(dyng::test::make_resources(b, threads));
    graphs.push_back(graph_t::from_edges(res.back(), edges.view(), props));
  }
  const std::int64_t n0 = graphs[0].num_vertices();
  const auto source = static_cast<vertex_t>(gen.uniform(0, n0 - 1));
  dyng::sssp::options opt;
  opt.objective = static_cast<int>(gen.uniform(0, gen.num_weights - 1));
  if (gen.coin(0.3)) {
    opt.delta = gen.uniform(1, 300);  // the width changes the schedule, never the result
  }
  std::vector<dyng::sssp::result<vertex_t>> results;
  for (std::size_t i = 0; i < backends.size(); ++i) {
    results.push_back(dyng::sssp::compute(res[i], graphs[i], source, opt));
    const auto check = dyng::testing::check_sssp_tree(graphs[i], results[i]);
    ASSERT_TRUE(check.ok()) << "compute on " << dyng::to_string(backends[i]) << ": "
                            << check.summary();
  }
  if (sc.perturb_ties) {
    const auto csr = graphs[0].to_csr(res[0]);
    const auto d = results[0].distances();
    const std::vector<std::int64_t> distances(d.begin(), d.end());
    const std::vector<vertex_t> parents = perturb_tie_parents(
        csr, opt.objective, source, dyng::host_view(distances), results[0].parents(), gen.rng);
    for (std::size_t i = 0; i < backends.size(); ++i) {
      results[i] = dyng::sssp::result<vertex_t>::from_arrays(
          res[i], graphs[i], source, dyng::host_view(distances), dyng::host_view(parents),
          /*canonicalize=*/false, opt);
    }
  }
  const int rounds = static_cast<int>(gen.uniform(3, 5));
  for (int round = 0; round < rounds; ++round) {
    SCOPED_TRACE("batch " + std::to_string(round));
    const auto csr = graphs[0].to_csr(res[0]);
    const auto p = results[0].parents();
    const std::vector<vertex_t> tree(p.begin(), p.end());
    const auto b = gen.batch(csr, tree, sc.growth, sc.mosp_compatible);
    std::vector<dyng::sssp::stats> st;
    for (std::size_t i = 0; i < backends.size(); ++i) {
      st.push_back(dyng::sssp::update(res[i], graphs[i], b.view(), results[i]));
      // A canonical input tree stays canonical; a perturbed one stays a valid shortest-path tree.
      const auto check = dyng::testing::check_sssp_tree(graphs[i], results[i],
                                                        /*require_canonical=*/!sc.perturb_ties);
      ASSERT_TRUE(check.ok()) << "update on " << dyng::to_string(backends[i]) << ": "
                              << check.summary();
      // update == compute on the new graph: bit for bit from a canonical tree; the distances
      // (and the parents in the distance-only mode) from a perturbed one.
      const auto fresh = dyng::sssp::compute(res[i], graphs[i], source, opt);
      const auto d = results[i].distances();
      const auto fd = fresh.distances();
      ASSERT_TRUE(std::equal(d.begin(), d.end(), fd.begin(), fd.end()));
      const auto pr = results[i].parents();
      const auto fp = fresh.parents();
      if (!sc.perturb_ties || !st[i].packed_parents) {
        ASSERT_TRUE(std::equal(pr.begin(), pr.end(), fp.begin(), fp.end()));
      }
      EXPECT_LE(st[i].affected, static_cast<std::int64_t>(graphs[i].num_vertices()));
      EXPECT_LE(st[i].invalidated, static_cast<std::int64_t>(graphs[i].num_vertices()));
      EXPECT_EQ(results[i].graph_version(), graphs[i].version());
    }
    // Cross-backend: identical trees and deterministic counters.
    for (std::size_t i = 1; i < backends.size(); ++i) {
      const auto d0 = results[0].distances();
      const auto di = results[i].distances();
      ASSERT_TRUE(std::equal(d0.begin(), d0.end(), di.begin(), di.end()));
      const auto p0 = results[0].parents();
      const auto pi = results[i].parents();
      ASSERT_TRUE(std::equal(p0.begin(), p0.end(), pi.begin(), pi.end()));
      EXPECT_EQ(st[0].invalidated, st[i].invalidated);
      EXPECT_EQ(st[0].affected, st[i].affected);
      EXPECT_EQ(st[0].batch.inserted_edges, st[i].batch.inserted_edges);
      EXPECT_EQ(st[0].batch.deleted_edges, st[i].batch.deleted_edges);
    }
  }
}

TYPED_TEST(SsspRandom, MospCompatibleGraphsMatchDijkstraOnEveryBackend) {
  for (const std::uint64_t seed : seeds(1000, 60)) {
    for (const shape s : all_shapes) {
      run_scenario<TypeParam>(seed, s, scenario{true, true, false});
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

TYPED_TEST(SsspRandom, NonCanonicalInputTreesAgreeOnEveryBackend) {
  for (const std::uint64_t seed : seeds(13000, 60)) {
    for (const shape s : all_shapes) {
      scenario sc{true, true, false};
      sc.perturb_ties = true;
      run_scenario<TypeParam>(seed, s, sc);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

TYPED_TEST(SsspRandom, SimpleSortedGraphsWithVertexGrowth) {
  for (const std::uint64_t seed : seeds(5000, 30)) {
    for (const shape s : all_shapes) {
      run_scenario<TypeParam>(seed, s, scenario{false, true, true});
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

TYPED_TEST(SsspRandom, UndirectedGraphs) {
  for (const std::uint64_t seed : seeds(9000, 20)) {
    for (const shape s : all_shapes) {
      run_scenario<TypeParam>(seed, s, scenario{false, false, true});
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

// Several results on one graph (dyng::update): the batch is applied once, and each result equals
// its own update on a copy of the graph.
TEST(SsspComposition, SeveralResultsOnOneGraphEqualSeparateUpdates) {
  using graph_t = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
  for (const std::uint64_t seed : seeds(20000, 40)) {
    SCOPED_TRACE("seed " + std::to_string(seed));
    generator<std::int32_t, std::int32_t> gen{std::mt19937_64(seed)};
    gen.num_weights = 3;
    gen.max_weight = 20;
    const auto edges = gen.graph_edges(all_shapes[seed % 5], true);
    for (const dyng::backend b : dyng::test::host_backends()) {
      const auto res = dyng::test::make_resources(b, 3);
      auto g = graph_t::from_edges(res, edges.view(), dyng::graph_properties::mosp_compatible());
      const auto n = static_cast<std::int32_t>(g.num_vertices());
      std::vector<dyng::sssp::options> opts(3);
      for (int k = 0; k < 3; ++k) {
        opts[static_cast<std::size_t>(k)].objective = k;
      }
      auto r0 = dyng::sssp::compute(res, g, 0, opts[0]);
      auto r1 = dyng::sssp::compute(res, g, n - 1, opts[1]);
      auto r2 = dyng::sssp::compute(res, g, n / 2, opts[2]);
      for (int round = 0; round < 3; ++round) {
        const auto p = r0.parents();
        const auto batch =
            gen.batch(g.to_csr(res), std::vector<std::int32_t>(p.begin(), p.end()), false, true);
        // Reference: each result updated alone on its own copy of the graph.
        std::vector<dyng::sssp::result<std::int32_t>> alone;
        std::vector<dyng::sssp::stats> alone_stats;
        for (auto* r : {&r0, &r1, &r2}) {
          auto copy = g.clone(res);
          auto rc = r->clone(res);
          alone_stats.push_back(dyng::sssp::update(res, copy, batch.view(), rc));
          alone.push_back(std::move(rc));
        }
        const std::uint64_t version = g.version();
        const auto [s0, s1, s2] = dyng::update(res, g, batch.view(), r0, r1, r2);
        EXPECT_EQ(g.version(), version + 1);  // applied once
        std::size_t i = 0;
        for (auto* r : {&r0, &r1, &r2}) {
          const auto d = r->distances();
          const auto da = alone[i].distances();
          EXPECT_TRUE(std::equal(d.begin(), d.end(), da.begin(), da.end()));
          const auto pr = r->parents();
          const auto pa = alone[i].parents();
          EXPECT_TRUE(std::equal(pr.begin(), pr.end(), pa.begin(), pa.end()));
          EXPECT_TRUE(dyng::testing::check_sssp_tree(g, *r).ok());
          ++i;
        }
        EXPECT_EQ(s0.invalidated, alone_stats[0].invalidated);
        EXPECT_EQ(s1.invalidated, alone_stats[1].invalidated);
        EXPECT_EQ(s2.invalidated, alone_stats[2].invalidated);
        EXPECT_EQ(s2.affected, alone_stats[2].affected);
      }
    }
  }
}

}  // namespace
