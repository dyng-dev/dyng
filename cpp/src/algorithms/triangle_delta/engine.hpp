// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file engine.hpp
 * @brief triangle_delta_engine_impl<exec_t>: the counts of triangle_delta, written once with the
 *        framework operators and run by the executor `exec_t` of a backend (TEACHING MATERIAL).
 *
 * Each count is a functor that does the work of ONE element (a vertex of the static count, a
 * changed edge of an update) and adds its triangles to one counter with an atomic addition. The
 * sorted intersection of two rows (`for_each_common`) is an operator with one user, so it lives
 * here and not in cpp/src/operators (the rule of two, PLAN 4.5.3).
 */
#pragma once

#include "algorithms/triangle_delta/problem.hpp"
#include "operators/execution.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace triangle_delta_ops {

using operators::atomic_add;

/**
 * @brief Call `f(w)` for every common neighbour w of u and v (a merge of the two sorted rows).
 * @tparam vertex_t   Vertex id type.
 * @tparam edge_t     Edge offset type.
 * @tparam function_t A callable `void(vertex_t)`.
 * @param[in] g The graph.
 * @param[in] u One vertex.
 * @param[in] v The other vertex.
 * @param[in] f Called once per common neighbour, in increasing order.
 */
template <typename vertex_t, typename edge_t, typename function_t>
DYNG_HD void for_each_common(const triangle_delta_graph<vertex_t, edge_t>& g, vertex_t u,
                             vertex_t v, function_t&& f) {
  edge_t a = g.offsets[u];
  edge_t b = g.offsets[v];
  const edge_t a_end = g.offsets[u + 1];
  const edge_t b_end = g.offsets[v + 1];
  while (a < a_end && b < b_end) {
    const vertex_t x = g.neighbors[a];
    const vertex_t y = g.neighbors[b];
    if (x < y) {
      ++a;
    } else if (y < x) {
      ++b;
    } else {
      f(x);
      ++a;
      ++b;
    }
  }
}

/**
 * @brief The static count: vertex u counts the triangles u < v < w (each triangle once, from its
 *        smallest vertex).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct count_from_vertex {
  triangle_delta_graph<vertex_t, edge_t> g;  ///< the graph
  std::uint64_t* total;                      ///< the count

  /// Vertex u.
  DYNG_HD void operator()(std::int64_t i) const {
    const auto u = static_cast<vertex_t>(i);
    std::uint64_t found = 0;
    for (edge_t e = g.offsets[u]; e < g.offsets[u + 1]; ++e) {
      const vertex_t v = g.neighbors[e];
      if (v <= u) {
        continue;
      }
      for_each_common(g, u, v, [&](vertex_t w) { found += w > v ? 1U : 0U; });
    }
    if (found != 0) {
      atomic_add(total, found);
    }
  }
};

/**
 * @brief Whether (a, b) (a < b) is one of the first `count` changed edges (a binary search of the
 *        sorted list: tails[0, count), heads[0, count)).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
DYNG_HD bool among_first(const vertex_t* tails, const vertex_t* heads, std::int64_t count,
                         vertex_t a, vertex_t b) {
  std::int64_t lo = 0;
  std::int64_t hi = count;
  while (lo < hi) {
    const std::int64_t mid = lo + (hi - lo) / 2;
    if (tails[mid] < a || (tails[mid] == a && heads[mid] < b)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < count && tails[lo] == a && heads[lo] == b;
}

/**
 * @brief The update count: changed edge i = (u, v) counts the triangles {u, v, w} it owns, those
 *        whose other two edges are not changed edges with a smaller id (ownership::min_member).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct count_owned_by_change {
  triangle_delta_graph<vertex_t, edge_t> g;  ///< G_t (deletions) or G_{t+1} (insertions)
  const vertex_t* tails;                     ///< the changed edges' smaller ends, sorted
  const vertex_t* heads;                     ///< their larger ends
  std::uint64_t* total;                      ///< the count

  /// Changed edge i.
  DYNG_HD void operator()(std::int64_t i) const {
    const vertex_t u = tails[i];
    const vertex_t v = heads[i];
    std::uint64_t found = 0;
    for_each_common(g, u, v, [&](vertex_t w) {
      if (w == u || w == v) {
        return;  // a self-loop is no triangle
      }
      // The edges (u, w) and (v, w), as (smaller, larger); owned by i unless one of them is a
      // changed edge with a smaller id.
      const bool smaller_owner =
          among_first(tails, heads, i, u < w ? u : w, u < w ? w : u) ||
          among_first(tails, heads, i, v < w ? v : w, v < w ? w : v);
      found += smaller_owner ? 0U : 1U;
    });
    if (found != 0) {
      atomic_add(total, found);
    }
  }
};

}  // namespace triangle_delta_ops

/**
 * @brief The counts of triangle_delta on the backend of `exec_t` (see the file comment and
 *        triangle_delta_engine in problem.hpp).
 * @tparam exec_t   The executor (operators::sequential_exec, openmp_exec or cuda_exec).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename exec_t, typename vertex_t, typename edge_t>
class triangle_delta_engine_impl final : public triangle_delta_engine<vertex_t, edge_t> {
 public:
  using graph_type = triangle_delta_graph<vertex_t, edge_t>;  ///< the graph
  using workspace_type = triangle_delta_workspace<vertex_t>;  ///< the workspace

  std::uint64_t count_all(const resources& res, const graph_type& g,
                          workspace_type& ws) const override {
    const exec_t exec(res);
    std::uint64_t* total = ws.counter.reserve(res, 1);
    exec.write(total, std::uint64_t{0});
    exec.for_each(g.num_vertices, triangle_delta_ops::count_from_vertex<vertex_t, edge_t>{g, total});
    return exec.read(total);
  }

  std::uint64_t count_owned(const resources& res, const graph_type& g,
                            const std::vector<std::pair<vertex_t, vertex_t>>& changes,
                            workspace_type& ws) const override {
    const auto count = static_cast<std::int64_t>(changes.size());
    if (count == 0) {
      return 0;  // nothing changed: no pass (and no host synchronization)
    }
    const exec_t exec(res);
    // The list as two arrays (tails, then heads) in the backend's memory.
    ws.staging.resize(2 * changes.size());
    for (std::size_t i = 0; i < changes.size(); ++i) {
      ws.staging[i] = changes[i].first;
      ws.staging[changes.size() + i] = changes[i].second;
    }
    vertex_t* lists = ws.changes.reserve(res, ws.staging.size());
    exec.upload(lists, ws.staging.data(), 2 * count);
    std::uint64_t* total = ws.counter.reserve(res, 1);
    exec.write(total, std::uint64_t{0});
    exec.for_each(count, triangle_delta_ops::count_owned_by_change<vertex_t, edge_t>{
                             g, lists, lists + count, total});
    return exec.read(total);
  }
};

}  // namespace dyng::detail
