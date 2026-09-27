// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/changeGenerator.cpp (generateChangeBatch, localBall,
// reachableFrom, filterUnsafeDeletions) and MOSP-CUDA@e220ee2:src/changeGenerator.cu (the same
// code without the saturation of "increase")
/**
 * @file legacy.cpp
 * @brief generators::legacy::mosp_changes(): MOSP's seeded change generator, bit-exact.
 *
 * Mechanical changes only: names, templates on the index types, exceptions instead of `cout` +
 * `false`, dynG's host transposition (equal to transposeCsrGraph, row order included), host apply
 * (applyChangeBatch under graph_properties::mosp_compatible()) and sssp::compute() on the
 * sequential backend (Dijkstra with lowest-id ties, equal to dijkstraCsrGraph) instead of MOSP's
 * functions, the draws of std::uniform_int_distribution<int> / <size_t> through
 * detail::legacy_uniform_int (the libstdc++ algorithm, so the stream does not depend on the
 * standard library), and a set of (u, v) pairs instead of the 64-bit key (u << 32) ^ v (the same
 * membership for 32-bit ids, exact for 64-bit ids).
 */
#include "graph/apply_host.hpp"
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/rng.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dyng::generators::legacy {

std::string mosp_change_report::summary() const {
  std::ostringstream out;
  out << "inserts=" << inserts << " reweights=" << reweights << " deletes=" << deletes;
  if (safe) {
    out << " (safe: kept " << deletes << " of " << requested_deletes << " after " << safe_rounds
        << " rounds)";
  }
  if (centre >= 0) {
    out << " local centre=" << centre << " hops=" << local_hops << " region=" << region
        << " vertices";
  }
  return out.str();
}

namespace {

/// Exact membership of (u, v) pairs (MOSP's unordered_set of edgeKey(u, v)).
template <typename vertex_t>
struct pair_hash {
  std::size_t operator()(const std::pair<vertex_t, vertex_t>& p) const noexcept {
    const auto u = static_cast<std::uint64_t>(p.first);
    const auto v = static_cast<std::uint64_t>(p.second);
    return std::hash<std::uint64_t>{}((u << 32U) ^ (v * 0x9E3779B97F4A7C15ULL));
  }
};

template <typename vertex_t>
using edge_set = std::unordered_set<std::pair<vertex_t, vertex_t>, pair_hash<vertex_t>>;

/// Vertices within `hops` of `centre`, ignoring edge direction (breadth-first, out-edges of the
/// graph first, then of the reverse graph, in row order).
template <typename vertex_t, typename edge_t, typename weight_t>
std::vector<vertex_t> local_ball(const csr<vertex_t, edge_t, weight_t>& graph,
                                 const csr<vertex_t, edge_t, weight_t>& reverse, vertex_t centre,
                                 std::int64_t hops) {
  std::vector<std::int64_t> level(static_cast<std::size_t>(graph.num_vertices()), -1);
  std::vector<vertex_t> ball{centre};
  level[static_cast<std::size_t>(centre)] = 0;
  for (std::size_t i = 0; i < ball.size(); ++i) {
    const vertex_t u = ball[i];
    if (level[static_cast<std::size_t>(u)] == hops) {
      continue;
    }
    for (const csr<vertex_t, edge_t, weight_t>* g : {&graph, &reverse}) {
      for (edge_t e = g->row_ptr[static_cast<std::size_t>(u)];
           e < g->row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
        const vertex_t w = g->col_ind[static_cast<std::size_t>(e)];
        if (level[static_cast<std::size_t>(w)] < 0) {
          level[static_cast<std::size_t>(w)] = level[static_cast<std::size_t>(u)] + 1;
          ball.push_back(w);
        }
      }
    }
  }
  return ball;
}

/// Vertices reachable from `source`.
template <typename vertex_t, typename edge_t, typename weight_t>
std::vector<char> reachable_from(const csr<vertex_t, edge_t, weight_t>& graph, vertex_t source) {
  std::vector<char> seen(static_cast<std::size_t>(graph.num_vertices()), 0);
  std::vector<vertex_t> queue{source};
  seen[static_cast<std::size_t>(source)] = 1;
  for (std::size_t i = 0; i < queue.size(); ++i) {
    const vertex_t u = queue[i];
    for (edge_t e = graph.row_ptr[static_cast<std::size_t>(u)];
         e < graph.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
      const vertex_t w = graph.col_ind[static_cast<std::size_t>(e)];
      if (seen[static_cast<std::size_t>(w)] == 0) {
        seen[static_cast<std::size_t>(w)] = 1;
        queue.push_back(w);
      }
    }
  }
  return seen;
}

/// Remove deletions until no vertex reachable from the source in the original graph becomes
/// unreachable (at most a few rounds).
template <typename vertex_t, typename edge_t, typename weight_t>
int filter_unsafe_deletions(const csr<vertex_t, edge_t, weight_t>& graph, vertex_t source,
                            edge_batch<vertex_t, weight_t>& batch) {
  const std::vector<char> before = reachable_from(graph, source);
  const auto n = static_cast<std::size_t>(graph.num_vertices());
  int rounds = 0;
  while (batch.num_deletions() > 0) {
    ++rounds;
    csr<vertex_t, edge_t, weight_t> updated;
    (void)detail::apply_batch_host(graph, batch.view(), graph_properties::mosp_compatible(),
                                   updated, static_cast<detail::apply_delta<vertex_t>*>(nullptr));
    const std::vector<char> after = reachable_from(updated, source);
    std::vector<char> lost(n, 0);
    bool any_lost = false;
    for (std::size_t v = 0; v < n; ++v) {
      if (before[v] != 0 && after[v] == 0) {
        lost[v] = 1;
        any_lost = true;
      }
    }
    if (!any_lost) {
      break;
    }
    edge_batch<vertex_t, weight_t> kept(batch.num_weights());
    for (std::size_t i = 0; i < batch.insert_src().size(); ++i) {
      const auto k = static_cast<std::size_t>(batch.num_weights());
      kept.insert_edge(batch.insert_src()[i], batch.insert_dst()[i],
                       array_view<const weight_t>(batch.insert_weights().data() + i * k, k));
    }
    std::size_t kept_deletes = 0;
    for (std::size_t i = 0; i < batch.num_deletions(); ++i) {
      const vertex_t u = batch.delete_src()[i];
      const vertex_t v = batch.delete_dst()[i];
      if (lost[static_cast<std::size_t>(u)] == 0 && lost[static_cast<std::size_t>(v)] == 0) {
        kept.delete_edge(u, v);
        ++kept_deletes;
      }
    }
    if (kept_deletes == batch.num_deletions()) {
      break;  // cannot happen: a lost vertex is an endpoint of a cut edge
    }
    batch = std::move(kept);
  }
  return rounds;
}

/// std::uniform_int_distribution<int>(a, b)(rng) of libstdc++.
template <typename rng_t>
std::int64_t draw(rng_t& rng, std::int64_t a, std::int64_t b) {
  return detail::legacy_uniform_int(rng, a, b);
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
edge_batch<vertex_t, weight_t> mosp_changes(const csr_view<vertex_t, edge_t, weight_t>& input,
                                            const mosp_change_options& options,
                                            mosp_change_report* report) try {
  // MOSP's CsrGraph: the CSR as it is (append order, parallel edges kept).
  csr<vertex_t, edge_t, weight_t> graph;
  detail::build_from_csr_host(input, graph_properties::mosp_compatible(), graph);
  const std::int64_t n = graph.num_vertices();
  const int num_objectives = graph.num_weights;
  const auto K = static_cast<std::size_t>(num_objectives);
  DYNG_EXPECTS(n >= 2 && num_objectives > 0 && options.num_changes >= 0 &&
                   options.weight_min <= options.weight_max && options.weight_min >= 1 &&
                   options.source >= 0 && options.source < n,
               "generators::legacy::mosp_changes: invalid change generator options (n = ", n,
               ", objectives ", num_objectives, ", changes ", options.num_changes, ", weights [",
               options.weight_min, ", ", options.weight_max, "], source ", options.source, ")");
  DYNG_EXPECTS(options.num_changes <= INT_MAX,
               "generators::legacy::mosp_changes: at most 2^31 - 1 changes (MOSP's int count)");

  const mosp_change_mode mode = options.mode;
  const bool splits_insert_delete =
      mode == mosp_change_mode::uniform || mode == mosp_change_mode::targeted;
  const bool tree_only = mode == mosp_change_mode::targeted || mode == mosp_change_mode::increase;
  std::int64_t insert_count = 0;
  std::int64_t delete_count = 0;
  std::int64_t reweight_count = 0;
  if (splits_insert_delete) {
    insert_count = static_cast<std::int64_t>(std::llround(static_cast<double>(options.num_changes) *
                                                          options.insertion_percentage / 100.0));
    insert_count = std::max<std::int64_t>(0, std::min(insert_count, options.num_changes));
    delete_count = options.num_changes - insert_count;
  } else {
    reweight_count = options.num_changes;
  }
#if !defined(__GLIBCXX__)
  if (tree_only && (delete_count > 0 || reweight_count > 0)) {
    throw not_supported_error(
        "generators::legacy::mosp_changes: the targeted and increase modes reproduce libstdc++'s "
        "std::shuffle; build dynG against libstdc++");
  }
#endif

  std::mt19937 rng(options.seed);
  auto weight_draw = [&] { return draw(rng, options.weight_min, options.weight_max); };

  // --- Candidate vertices (all, or a local ball) -----------------------------------------------
  std::vector<vertex_t> region;
  std::vector<char> in_region;
  std::int64_t centre = -1;
  if (options.local_hops > 0) {
    csr<vertex_t, edge_t, weight_t> reverse;
    detail::transpose_host(graph, reverse);
    centre = draw(rng, 0, n - 1);
    region = local_ball(graph, reverse, static_cast<vertex_t>(centre), options.local_hops);
    in_region.assign(static_cast<std::size_t>(n), 0);
    for (const vertex_t v : region) {
      in_region[static_cast<std::size_t>(v)] = 1;
    }
    DYNG_EXPECTS(region.size() >= 2,
                 "generators::legacy::mosp_changes: the local region around "
                 "vertex ",
                 centre, " has fewer than 2 vertices");
  }
  auto random_vertex = [&]() -> vertex_t {
    if (region.empty()) {
      return static_cast<vertex_t>(draw(rng, 0, n - 1));
    }
    return region[static_cast<std::size_t>(
        draw(rng, 0, static_cast<std::int64_t>(region.size()) - 1))];
  };

  // --- Existing edges eligible for deletion / re-weighting --------------------------------------
  // In the order of the original generator: row order, first occurrence of each (u,v), no
  // self-loops (or the tree edges, vertex by vertex, tree by tree).
  auto collect_edges = [&](const std::vector<std::vector<vertex_t>>& trees) {
    std::vector<std::pair<vertex_t, vertex_t>> edges;
    edge_set<vertex_t> seen;
    if (tree_only) {
      for (std::int64_t v = 0; v < n; ++v) {
        if (!in_region.empty() && in_region[static_cast<std::size_t>(v)] == 0) {
          continue;
        }
        for (const auto& tree : trees) {
          const vertex_t p = tree[static_cast<std::size_t>(v)];
          if (p >= 0 && seen.insert({p, static_cast<vertex_t>(v)}).second) {
            edges.push_back({p, static_cast<vertex_t>(v)});
          }
        }
      }
      return edges;
    }
    for (std::int64_t u = 0; u < n; ++u) {
      if (!in_region.empty() && in_region[static_cast<std::size_t>(u)] == 0) {
        continue;
      }
      for (edge_t e = graph.row_ptr[static_cast<std::size_t>(u)];
           e < graph.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
        const vertex_t v = graph.col_ind[static_cast<std::size_t>(e)];
        if (u == static_cast<std::int64_t>(v) ||
            (!in_region.empty() && in_region[static_cast<std::size_t>(v)] == 0)) {
          continue;
        }
        if (seen.insert({static_cast<vertex_t>(u), v}).second) {
          edges.push_back({static_cast<vertex_t>(u), v});
        }
      }
    }
    return edges;
  };

  // The SOSP trees (dijkstraCsrGraph: Dijkstra with lowest-id ties) of every objective.
  std::vector<std::vector<vertex_t>> trees;
  if (tree_only) {
    const resources seq = resources::sequential();
    const auto g = dyng::graph<vertex_t, edge_t, weight_t>::from_csr(
        seq, graph.view(), graph_properties::mosp_compatible());
    for (int k = 0; k < num_objectives; ++k) {
      sssp::options opt;
      opt.objective = k;
      const auto tree = sssp::compute(seq, g, static_cast<vertex_t>(options.source), opt);
      const array_view<const vertex_t> parents = tree.parents();
      trees.emplace_back(parents.begin(), parents.end());
    }
  }

  // --- Insertions --------------------------------------------------------------------------------
  std::vector<std::pair<vertex_t, vertex_t>> insert_edges;
  insert_edges.reserve(static_cast<std::size_t>(insert_count));
  while (static_cast<std::int64_t>(insert_edges.size()) < insert_count) {
    const vertex_t u = random_vertex();
    const vertex_t v = random_vertex();
    if (u != v) {
      insert_edges.push_back({u, v});
    }
  }

  // --- Deletions / re-weighted edges -------------------------------------------------------------
  std::vector<std::pair<vertex_t, vertex_t>> delete_edges;
  std::vector<std::pair<vertex_t, vertex_t>> reweight_edges;
  if (delete_count > 0 || reweight_count > 0) {
    std::vector<std::pair<vertex_t, vertex_t>> candidates = collect_edges(trees);
    const std::int64_t wanted = std::max(delete_count, reweight_count);
    DYNG_EXPECTS(!candidates.empty(),
                 "generators::legacy::mosp_changes: no existing edges available for the batch");
    std::vector<std::pair<vertex_t, vertex_t>>& chosen =
        delete_count > 0 ? delete_edges : reweight_edges;
    if (mode == mosp_change_mode::uniform || mode == mosp_change_mode::reweight) {
      // Sampling with replacement, as generateChangedEdges(duplicate=true).
      const auto last = static_cast<std::int64_t>(candidates.size()) - 1;
      for (std::int64_t i = 0; i < wanted; ++i) {
        chosen.push_back(candidates[static_cast<std::size_t>(draw(rng, 0, last))]);
      }
    } else {
#if defined(__GLIBCXX__)
      // Distinct tree edges (libstdc++'s std::shuffle, as MOSP's generator draws them).
      std::shuffle(candidates.begin(), candidates.end(), rng);
#endif
      chosen.assign(candidates.begin(),
                    candidates.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(
                                             candidates.size(), static_cast<std::size_t>(wanted))));
    }
  }

  // --- Weights ------------------------------------------------------------------------------------
  const std::size_t m = graph.col_ind.size();
  std::vector<std::int64_t> below_average(K, options.weight_max);
  if (mode == mosp_change_mode::targeted && m > 0) {
    for (std::size_t k = 0; k < K; ++k) {
      double sum = 0;
      for (std::size_t e = 0; e < m; ++e) {
        sum += static_cast<double>(graph.weights[k * m + e]);
      }
      const auto average =
          static_cast<std::int64_t>(static_cast<int>(sum / static_cast<double>(m)));
      below_average[k] = std::max<std::int64_t>(options.weight_min, average - 1);
    }
  }
  edge_batch<vertex_t, weight_t> batch(num_objectives);
  batch.reserve(insert_edges.size() + reweight_edges.size(), delete_edges.size());
  std::vector<weight_t> w(K);
  for (const auto& [u, v] : insert_edges) {
    for (std::size_t k = 0; k < K; ++k) {
      w[k] = static_cast<weight_t>(mode == mosp_change_mode::targeted
                                       ? draw(rng, options.weight_min, below_average[k])
                                       : weight_draw());
    }
    batch.insert_edge(u, v, array_view<const weight_t>(w.data(), K));
  }
  for (const auto& [u, v] : reweight_edges) {
    auto e = static_cast<std::size_t>(graph.row_ptr[static_cast<std::size_t>(u)]);
    while (graph.col_ind[e] != v) {
      ++e;
    }
    for (std::size_t k = 0; k < K; ++k) {
      if (mode == mosp_change_mode::increase) {
        // Increases saturate at the largest valid weight (2^31 - 1), as MOSP-OpenMP's.
        w[k] = static_cast<weight_t>(
            std::min<std::int64_t>(INT_MAX, static_cast<std::int64_t>(graph.weights[k * m + e]) +
                                                draw(rng, 1, options.weight_max)));
      } else {
        w[k] = static_cast<weight_t>(weight_draw());
      }
    }
    batch.insert_edge(u, v, array_view<const weight_t>(w.data(), K));
  }
  for (const auto& [u, v] : delete_edges) {
    batch.delete_edge(u, v);
  }

  int safe_rounds = 0;
  const auto requested_deletes = static_cast<std::int64_t>(batch.num_deletions());
  if (options.safe_deletions) {
    safe_rounds = filter_unsafe_deletions(graph, static_cast<vertex_t>(options.source), batch);
  }
  if (report != nullptr) {
    mosp_change_report r;
    r.inserts = static_cast<std::int64_t>(batch.num_insertions() - reweight_edges.size());
    r.reweights = static_cast<std::int64_t>(reweight_edges.size());
    r.deletes = static_cast<std::int64_t>(batch.num_deletions());
    r.requested_deletes = requested_deletes;
    r.safe = options.safe_deletions;
    r.safe_rounds = safe_rounds;
    r.centre = centre;
    r.local_hops = centre >= 0 ? options.local_hops : 0;
    r.region = static_cast<std::int64_t>(region.size());
    *report = r;
  }
  return batch;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::mosp_changes (", input.num_vertices(),
                                  " vertices, ", options.num_changes, " changes)")

#define DYNG_INSTANTIATE_LEGACY(V, E, W)           \
  template edge_batch<V, W> mosp_changes<V, E, W>( \
      const csr_view<V, E, W>&, const mosp_change_options&, mosp_change_report*);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_LEGACY)
#undef DYNG_INSTANTIATE_LEGACY

}  // namespace dyng::generators::legacy
