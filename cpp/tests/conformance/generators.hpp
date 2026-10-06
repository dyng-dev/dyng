// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file generators.hpp
 * @brief The graphs and batches of the conformance kit (PLAN Section 8.2, C2): seeded random
 *        directed (or undirected) graphs kept in a host model, and batch mixes generated from the
 *        model so that every batch is valid under both batch-semantics presets and has a known
 *        inverse (C5).
 */
#pragma once

#include "conformance/test_traits.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::conformance {

/// The batch mixes of C2 (PLAN Section 8.2).
enum class batch_mix : std::uint8_t {
  insert_only,  ///< insertions of absent edges
  delete_only,  ///< deletions of existing edges
  mixed,        ///< half deletions, half insertions
  local,        ///< changes around a window of eight vertices
  heavy,        ///< 40 % of the edges deleted and as many inserted
  reweight,     ///< new weights of existing edges (upsert semantics, weighted graphs only)
  grow,         ///< insertions that name new vertices (vertex growth)
};

/// The mix's name (for traces).
inline std::string_view to_string(batch_mix mix) noexcept {
  switch (mix) {
    case batch_mix::insert_only:
      return "insert_only";
    case batch_mix::delete_only:
      return "delete_only";
    case batch_mix::mixed:
      return "mixed";
    case batch_mix::local:
      return "local";
    case batch_mix::heavy:
      return "heavy";
    case batch_mix::reweight:
      return "reweight";
    case batch_mix::grow:
      return "grow";
  }
  return "?";
}

/// Every mix, in the order C2 runs them.
inline std::vector<batch_mix> all_mixes() {
  return {batch_mix::insert_only, batch_mix::delete_only, batch_mix::mixed, batch_mix::local,
          batch_mix::heavy,       batch_mix::reweight,    batch_mix::grow};
}

/**
 * @brief A host model of a simple directed graph (no self-loops, no parallel edges): the edges
 *        and their weights (ignored for unweighted graphs).
 *
 * An undirected model (`directed` false, for an algorithm whose requirements set
 * graph_properties::directed = false) keeps every edge once, as the pair (u, v) with u < v; the
 * graph stores it in both directions and counts every batch edge once per direction
 * (apply_summary).
 *
 * A weighted model has `num_weights` columns (test_traits::num_weights, default 1): the model
 * stores the first column, and every further column is a fixed function of the edge and the
 * first column's weight (column_weight()), so a batch that changes an edge's weight changes the
 * other columns too, some up and some down (multi-objective algorithms see per-column increases
 * and decreases).
 * @tparam graph_t The graph type.
 */
template <typename graph_t>
struct graph_model {
  using vertex_type = typename graph_t::vertex_type;  ///< vertex ids
  using weight_type = typename graph_t::weight_type;  ///< weights (or unweighted)
  using edge = std::pair<vertex_type, vertex_type>;   ///< (u, v)
  /// Whether the graph has a weight column.
  static constexpr bool weighted = !is_unweighted_v<weight_type>;
  /// The largest number of weight columns of a model.
  static constexpr int max_columns = 64;

  vertex_type num_vertices = 0;          ///< n
  std::map<edge, std::int32_t> weights;  ///< the edges and their weights (column 0)
  int max_weight = 9;                    ///< new weights are drawn from [1, max_weight]
  int num_weights = 1;                   ///< weight columns of a weighted graph
  bool directed = true;                  ///< false: an undirected model (pairs u < v)

  /// The key of the edge (u, v) in `weights`: (u, v), or (min, max) for an undirected model.
  [[nodiscard]] edge key(vertex_type u, vertex_type v) const noexcept {
    return directed || u < v ? edge{u, v} : edge{v, u};
  }

  /// Whether the model has the edge (u, v) (in either direction for an undirected model).
  [[nodiscard]] bool has(vertex_type u, vertex_type v) const {
    return weights.count(key(u, v)) != 0;
  }

  /// The stored directions of one edge (1, or 2 for an undirected model): the factor of the
  /// edge count and of the apply_summary counters of a batch without self-loops.
  [[nodiscard]] std::int64_t directions() const noexcept {
    return directed ? 1 : 2;
  }

  /// The weight of column `k` of the edge (u, v) whose column 0 holds `w` (in [1, max_weight]).
  [[nodiscard]] std::int32_t column_weight(vertex_type u, vertex_type v, std::int32_t w,
                                           int k) const {
    if (k == 0) {
      return w;
    }
    const auto h = static_cast<std::uint64_t>(w) * 2654435761ULL +
                   static_cast<std::uint64_t>(k) * 40503ULL +
                   static_cast<std::uint64_t>(u) * 97ULL + static_cast<std::uint64_t>(v) * 31ULL;
    return static_cast<std::int32_t>(1 + (h >> 7) % static_cast<std::uint64_t>(max_weight));
  }

  /// The graph of the model on `res` with `props`.
  [[nodiscard]] graph_t build(const resources& res, graph_properties props) const {
    edge_list<vertex_type, weight_type> list;
    list.num_vertices = num_vertices;
    list.num_weights = weighted ? num_weights : 0;
    for (const auto& [e, w] : weights) {
      list.src.push_back(e.first);
      list.dst.push_back(e.second);
      if constexpr (weighted) {
        for (int k = 0; k < num_weights; ++k) {
          list.weights.push_back(static_cast<weight_type>(column_weight(e.first, e.second, w, k)));
        }
      } else {
        (void)w;
      }
    }
    props.num_weights = weighted ? num_weights : 0;
    return graph_t::from_edges(res, list.view(), props);
  }
};

/// A random model of `shape` (distinct edges without self-loops); undirected if `directed` is
/// false.
template <typename graph_t>
graph_model<graph_t> random_model(const graph_shape& shape, std::mt19937_64& rng,
                                  bool directed = true) {
  using vertex_type = typename graph_t::vertex_type;
  graph_model<graph_t> m;
  m.num_vertices = static_cast<vertex_type>(shape.vertices);
  m.max_weight = shape.max_weight;
  m.directed = directed;
  const std::int64_t n = shape.vertices;
  const std::int64_t possible = directed ? n * (n - 1) : n * (n - 1) / 2;
  const std::int64_t wanted = std::min(shape.edges, possible);
  std::uniform_int_distribution<std::int64_t> vertex(0, std::max<std::int64_t>(n - 1, 0));
  std::uniform_int_distribution<int> weight(1, shape.max_weight);
  while (static_cast<std::int64_t>(m.weights.size()) < wanted) {
    const auto u = static_cast<vertex_type>(vertex(rng));
    const auto v = static_cast<vertex_type>(vertex(rng));
    if (u != v) {
      m.weights.emplace(m.key(u, v), weight(rng));
    }
  }
  return m;
}

/**
 * @brief One generated batch, its effect on the model and its inverse (C5).
 * @tparam graph_t The graph type.
 */
template <typename graph_t>
struct generated_batch {
  using vertex_type = typename graph_t::vertex_type;  ///< vertex ids
  using weight_type = typename graph_t::weight_type;  ///< weights
  edge_batch<vertex_type, weight_type> batch;         ///< the batch
  edge_batch<vertex_type, weight_type> inverse;       ///< undoes it (no vertex growth only)
  bool invertible = true;                             ///< false: the batch grows the vertex set
};

namespace generators_detail {

template <typename graph_t>
void insert(const graph_model<graph_t>& m,
            edge_batch<typename graph_t::vertex_type, typename graph_t::weight_type>& b,
            typename graph_t::vertex_type u, typename graph_t::vertex_type v, std::int32_t w) {
  using weight_type = typename graph_t::weight_type;
  if constexpr (graph_model<graph_t>::weighted) {
    weight_type columns[graph_model<graph_t>::max_columns] = {};
    for (int k = 0; k < m.num_weights; ++k) {
      columns[k] = static_cast<weight_type>(m.column_weight(u, v, w, k));
    }
    b.insert_edge(u, v,
                  array_view<const weight_type>(columns, static_cast<std::size_t>(m.num_weights)));
  } else {
    (void)m;
    (void)w;
    b.insert_edge(u, v);
  }
}

}  // namespace generators_detail

/**
 * @brief A batch of `mix` from the model, applied to the model (the caller applies it to the
 *        graph). Every batch is valid under batch_semantics::upsert_last_wins() and set(): its
 *        deletions name existing edges, its insertions absent ones (reweight: existing ones, so
 *        callers use it under upsert semantics only), with no duplicates and no self-loops.
 * @tparam graph_t The graph type.
 * @param[in,out] m   The model.
 * @param[in]     mix The mix.
 * @param[in,out] rng The generator.
 * @return The batch and its inverse.
 */
template <typename graph_t>
generated_batch<graph_t> random_batch(graph_model<graph_t>& m, batch_mix mix,
                                      std::mt19937_64& rng) {
  using vertex_type = typename graph_t::vertex_type;
  using edge = typename graph_model<graph_t>::edge;
  generated_batch<graph_t> out;
  if constexpr (graph_model<graph_t>::weighted) {
    out.batch = edge_batch<vertex_type, typename graph_t::weight_type>(m.num_weights);
    out.inverse = edge_batch<vertex_type, typename graph_t::weight_type>(m.num_weights);
  }
  const std::int64_t n = m.num_vertices;
  const auto m_edges = static_cast<std::int64_t>(m.weights.size());
  const std::int64_t k = std::max<std::int64_t>(1, m_edges / 10);
  std::uniform_int_distribution<int> weight(1, m.max_weight);
  // The window of the local mix.
  std::uniform_int_distribution<std::int64_t> start(0, std::max<std::int64_t>(n - 8, 0));
  const std::int64_t w0 = start(rng);
  const std::int64_t w1 = std::min(n, w0 + 8);
  const auto in_window = [&](const edge& e) {
    return (e.first >= w0 && e.first < w1) || (e.second >= w0 && e.second < w1);
  };
  std::int64_t deletions = 0;
  std::int64_t insertions = 0;
  std::int64_t reweights = 0;
  switch (mix) {
    case batch_mix::insert_only:
      insertions = k;
      break;
    case batch_mix::delete_only:
      deletions = k;
      break;
    case batch_mix::mixed:
    case batch_mix::local:
      deletions = (k + 1) / 2;
      insertions = (k + 1) / 2;
      break;
    case batch_mix::heavy:
      deletions = std::max<std::int64_t>(1, m_edges * 2 / 5);
      insertions = deletions;
      break;
    case batch_mix::reweight:
      reweights = k;
      deletions = (k + 3) / 4;
      break;
    case batch_mix::grow:
      insertions = std::max<std::int64_t>(2, k / 2);
      break;
  }
  // Deletions (and reweights) of existing edges.
  std::vector<edge> existing;
  existing.reserve(m.weights.size());
  for (const auto& entry : m.weights) {
    if (mix != batch_mix::local || in_window(entry.first)) {
      existing.push_back(entry.first);
    }
  }
  std::shuffle(existing.begin(), existing.end(), rng);
  std::size_t next = 0;
  for (std::int64_t i = 0; i < deletions && next < existing.size(); ++i, ++next) {
    const edge e = existing[next];
    out.batch.delete_edge(e.first, e.second);
    generators_detail::insert<graph_t>(m, out.inverse, e.first, e.second, m.weights.at(e));
    m.weights.erase(e);
  }
  std::vector<std::pair<edge, std::int32_t>> old_weights;
  for (std::int64_t i = 0; i < reweights && next < existing.size(); ++i, ++next) {
    const edge e = existing[next];
    const std::int32_t w = weight(rng);
    generators_detail::insert<graph_t>(m, out.batch, e.first, e.second, w);
    old_weights.emplace_back(e, m.weights.at(e));
    m.weights[e] = w;
  }
  // Insertions of absent edges (grow: from existing vertices to new ones).
  std::vector<edge> inserted;
  if (mix == batch_mix::grow) {
    const std::int64_t added = 1 + static_cast<std::int64_t>(rng() % 3);
    std::uniform_int_distribution<std::int64_t> old_vertex(0, std::max<std::int64_t>(n - 1, 0));
    std::int64_t grown = n;
    for (std::int64_t i = 0; i < insertions; ++i) {
      const auto fresh = static_cast<vertex_type>(n + (i % added));
      const auto old = static_cast<vertex_type>(old_vertex(rng));
      const edge e = (i % 2 == 0) ? edge{old, fresh} : m.key(fresh, old);
      if (e.first != e.second && m.weights.emplace(e, weight(rng)).second) {
        generators_detail::insert<graph_t>(m, out.batch, e.first, e.second, m.weights.at(e));
        grown = std::max<std::int64_t>(grown, static_cast<std::int64_t>(fresh) + 1);
      }
    }
    m.num_vertices = static_cast<vertex_type>(grown);
    out.invertible = false;
  } else if (n > 1) {
    std::uniform_int_distribution<std::int64_t> vertex(mix == batch_mix::local ? w0 : 0,
                                                       mix == batch_mix::local ? w1 - 1 : n - 1);
    std::uniform_int_distribution<std::int64_t> any(0, n - 1);
    const std::int64_t possible = m.directed ? n * (n - 1) : n * (n - 1) / 2;
    for (std::int64_t i = 0, tries = 0;
         i < insertions && static_cast<std::int64_t>(m.weights.size()) < possible && tries < 64 * k;
         ++tries) {
      const auto x = static_cast<vertex_type>(vertex(rng));
      const auto y = static_cast<vertex_type>(tries % 2 == 0 ? any(rng) : vertex(rng));
      const edge e = m.key(x, y);
      const vertex_type u = e.first;
      const vertex_type v = e.second;
      if (u != v && m.weights.count(e) == 0) {
        // An edge deleted by this batch is not re-inserted (that would be a cancelled pair).
        bool deleted = false;
        for (std::size_t j = 0; j < out.batch.num_deletions(); ++j) {
          deleted = deleted || (out.batch.delete_src()[j] == u && out.batch.delete_dst()[j] == v);
        }
        if (!deleted) {
          m.weights.emplace(e, weight(rng));
          generators_detail::insert<graph_t>(m, out.batch, u, v, m.weights.at(e));
          inserted.push_back(e);
          ++i;
        }
      }
    }
  }
  for (const edge& e : inserted) {
    out.inverse.delete_edge(e.first, e.second);
  }
  for (const auto& [e, w] : old_weights) {
    generators_detail::insert<graph_t>(m, out.inverse, e.first, e.second, w);
  }
  return out;
}

}  // namespace dyng::conformance
