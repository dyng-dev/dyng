// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/combinedGraphCpu.cpp (combinedEdge, combinedGraphSospCpu
// Step 2)
/**
 * @file openmp.cpp
 * @brief The OpenMP backend of mosp's combine step: MOSP-OpenMP's combinedGraphSospCpu, Step 2
 *        (one loop iteration per vertex compares its K parents; relaxed atomic out-degree counts,
 *        prefix sum, parallel fill with relaxed atomic cursors).
 *
 * Mechanical changes: templates on the index types; the K parent arrays of the sssp results
 * instead of one objective-major array; the thread count of the resources handle; the vectors of
 * the pooled workspace. The fill order inside a row depends on the schedule, which the solve does
 * not see (lowest-id ties).
 */
#include "algorithms/mosp/problem.hpp"
#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
#include "graph/instantiate.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng::detail {

#if DYNG_HAS_OPENMP

namespace {

/// MOSP's combinedEdge() (see sequential.cpp).
template <typename vertex_t, typename weight_t>
inline bool combined_edge(const mosp_combine_input<vertex_t>& in, vertex_t v, int k, vertex_t& p,
                          weight_t& weight) {
  p = in.parents[k][v];
  if (p < 0) {
    return false;
  }
  for (int j = 0; j < k; ++j) {
    if (in.parents[j][v] == p) {
      return false;  // counted with its first occurrence
    }
  }
  std::int32_t w = in.base - in.terms[k];
  for (int j = k + 1; j < in.num_objectives; ++j) {
    if (in.parents[j][v] == p) {
      w -= in.terms[j];
    }
  }
  weight = static_cast<weight_t>(w);
  return true;
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_openmp(
    const resources& res, const mosp_combine_input<vertex_t>& in,
    mosp_workspace<vertex_t, edge_t, weight_t>& ws) {
  const int threads = resources_access::host_threads(res);
  const auto n = static_cast<std::int64_t>(in.num_vertices);
  const vertex_t source = in.source;
  const int num_k = in.num_objectives;

  // Step 2: count out-degrees, prefix sum, fill.
  ws.cursor.assign(static_cast<std::size_t>(n) + 1, edge_t{0});
  std::int64_t edges = 0;
  std::int64_t weight_sum = 0;
  edge_t* degree = ws.cursor.data();
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : edges, weight_sum)
  for (std::int64_t v = 0; v < n; ++v) {
    if (v == static_cast<std::int64_t>(source)) {
      continue;
    }
    for (int k = 0; k < num_k; ++k) {
      vertex_t p = 0;
      weight_t weight = 0;
      if (combined_edge(in, static_cast<vertex_t>(v), k, p, weight)) {
        __atomic_fetch_add(&degree[p], edge_t{1}, __ATOMIC_RELAXED);
        ++edges;
        weight_sum += weight;
      }
    }
  }
  ws.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  ws.row_ptr[0] = 0;
  std::partial_sum(ws.cursor.begin(), ws.cursor.end() - 1, ws.row_ptr.begin() + 1);
  std::copy(ws.row_ptr.begin(), ws.row_ptr.end() - 1, ws.cursor.begin());
  ws.col_ind.resize(static_cast<std::size_t>(edges));
  ws.weights.resize(static_cast<std::size_t>(edges));
  edge_t* cursor = ws.cursor.data();
  vertex_t* col_ind = ws.col_ind.data();
  weight_t* weights = ws.weights.data();
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t v = 0; v < n; ++v) {
    if (v == static_cast<std::int64_t>(source)) {
      continue;
    }
    for (int k = 0; k < num_k; ++k) {
      vertex_t p = 0;
      weight_t weight = 0;
      if (combined_edge(in, static_cast<vertex_t>(v), k, p, weight)) {
        const edge_t position = __atomic_fetch_add(&cursor[p], edge_t{1}, __ATOMIC_RELAXED);
        col_ind[position] = static_cast<vertex_t>(v);
        weights[position] = weight;
      }
    }
  }

  mosp_combined<vertex_t, edge_t, weight_t> out;
  out.view.num_vertices = in.num_vertices;
  out.view.out_row_ptr = ws.row_ptr.data();
  out.view.out_col_ind = ws.col_ind.data();
  out.view.out_weights = ws.weights.data();
  out.edges = edges;
  out.weight_sum = weight_sum;
  return out;
}

namespace {

/// The level state of the parallel traversal, written by one thread between two barriers.
struct path_level {
  std::int64_t begin = 0;  ///< first queue position of the level
  std::int64_t end = 0;    ///< one past its last
  bool stop = false;       ///< a tree edge is missing: stop (every thread reads the same value)
};

/// Below this many vertices the sequential path costs are used (the fork costs more).
constexpr std::int64_t parallel_path_cost_vertices = std::int64_t{1} << 14;

/// A level narrower than this many nodes per thread is walked by one thread.
constexpr std::int64_t nodes_per_thread = 8;

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t mosp_path_costs_openmp(const resources& res,
                                const csr_view<vertex_t, edge_t, weight_t>& out,
                                const vertex_t* parent, vertex_t source, int k, std::int64_t* costs,
                                mosp_workspace<vertex_t, edge_t, weight_t>& ws) {
  const int threads = resources_access::host_threads(res);
  const auto n = static_cast<std::int64_t>(out.num_vertices());
  if (threads <= 1 || n < parallel_path_cost_vertices) {
    return mosp_path_costs(out, parent, source, k, costs, ws);
  }
  const auto num_k = static_cast<std::size_t>(k);
  const auto rows = static_cast<std::size_t>(n) + 1;
  // Within the reserved capacities (mosp_workspace::reserve); the per-thread sums grow once.
  ws.child_start.resize(rows);
  ws.children.resize(static_cast<std::size_t>(n));
  ws.queue.resize(static_cast<std::size_t>(n));
  if (ws.thread_sums.size() < static_cast<std::size_t>(threads) + 1) {
    note_reservation();  // invariant I9: a reserving run (core/budget_counters.hpp)
    ws.thread_sums.resize(static_cast<std::size_t>(threads) + 1);
  }
  vertex_t* child_start = ws.child_start.data();
  vertex_t* children = ws.children.data();
  vertex_t* queue = ws.queue.data();
  std::int64_t* sums = ws.thread_sums.data();
  const edge_t* row_ptr = out.row_ptr.data();
  const vertex_t* col_ind = out.col_ind.data();
  const weight_t* weights = out.weights.data();
  const auto m = static_cast<std::size_t>(out.col_ind.size());
  const auto total = static_cast<std::int64_t>(static_cast<std::size_t>(n) * num_k);
  const auto s = static_cast<std::int64_t>(source);
  path_level levels[2];  // level i is levels[i % 2]; the next one is written into the other
  bool missing = false;  // set (relaxed) by any thread that finds a tree edge missing

  // The costs of v from those of its parent p over the first edge p -> v of p's row
  // (mospPathCosts); false if there is no such edge.
  const auto relax = [&](vertex_t p, vertex_t v) {
    edge_t edge = -1;
    for (edge_t e = row_ptr[p]; e < row_ptr[p + 1]; ++e) {
      if (col_ind[e] == v) {
        edge = e;
        break;
      }
    }
    if (edge < 0) {
      __atomic_store_n(&missing, true, __ATOMIC_RELAXED);
      return false;
    }
    for (std::size_t j = 0; j < num_k; ++j) {
      costs[static_cast<std::size_t>(v) * num_k + j] =
          costs[static_cast<std::size_t>(p) * num_k + j] +
          static_cast<std::int64_t>(weights[j * m + static_cast<std::size_t>(edge)]);
    }
    return true;
  };

#pragma omp parallel num_threads(threads)
  {
    const auto t = static_cast<std::int64_t>(omp_get_thread_num());
    const auto team = static_cast<std::int64_t>(omp_get_num_threads());
#pragma omp for schedule(static) nowait
    for (std::int64_t i = 0; i < total; ++i) {
      costs[i] = sssp_infinity;
    }
#pragma omp for schedule(static)
    for (std::int64_t v = 0; v <= n; ++v) {
      child_start[v] = 0;
    }
    // Children lists of the tree: counts, prefix sum (one range per thread), atomic fills.
#pragma omp for schedule(static)
    for (std::int64_t v = 0; v < n; ++v) {
      if (v != s && parent[v] >= 0) {
        __atomic_fetch_add(&child_start[parent[v] + 1], vertex_t{1}, __ATOMIC_RELAXED);
      }
    }
    const std::int64_t lo = 1 + n * t / team;
    const std::int64_t hi = 1 + n * (t + 1) / team;
    std::int64_t local = 0;
    for (std::int64_t v = lo; v < hi; ++v) {
      local += child_start[v];
    }
    sums[t + 1] = local;
#pragma omp barrier
#pragma omp single
    {
      sums[0] = 0;
      for (std::int64_t i = 0; i < team; ++i) {
        sums[i + 1] += sums[i];
      }
    }
    std::int64_t running = sums[t];
    for (std::int64_t v = lo; v < hi; ++v) {
      running += child_start[v];
      child_start[v] = static_cast<vertex_t>(running);
    }
#pragma omp barrier
    // The fill cursor uses `queue` (the traversal overwrites it).
#pragma omp for schedule(static)
    for (std::int64_t v = 0; v < n; ++v) {
      queue[v] = child_start[v];
    }
#pragma omp for schedule(static)
    for (std::int64_t v = 0; v < n; ++v) {
      if (v != s && parent[v] >= 0) {
        const vertex_t position =
            __atomic_fetch_add(&queue[parent[v]], vertex_t{1}, __ATOMIC_RELAXED);
        children[position] = static_cast<vertex_t>(v);
      }
    }
    // The traversal from the source, one level per iteration.
#pragma omp single
    {
      for (std::size_t j = 0; j < num_k; ++j) {
        costs[static_cast<std::size_t>(source) * num_k + j] = 0;
      }
      queue[0] = source;
      levels[0] = path_level{0, 1, false};
    }
    for (int level = 0;; level ^= 1) {
      const path_level here = levels[level];  // written before the last barrier
      if (here.stop || here.begin >= here.end) {
        break;
      }
      path_level& next = levels[level ^ 1];
      const std::int64_t width = here.end - here.begin;
      if (width < nodes_per_thread * team) {
#pragma omp single
        {
          std::int64_t end = here.end;
          for (std::int64_t i = here.begin; i < here.end; ++i) {
            const vertex_t p = queue[i];
            for (vertex_t c = child_start[p]; c < child_start[p + 1]; ++c) {
              const vertex_t v = children[c];
              (void)relax(p, v);
              queue[end++] = v;
            }
          }
          next = path_level{here.end, end, __atomic_load_n(&missing, __ATOMIC_RELAXED)};
        }
      } else {
        const std::int64_t first = here.begin + width * t / team;
        const std::int64_t last = here.begin + width * (t + 1) / team;
        std::int64_t count = 0;
        for (std::int64_t i = first; i < last; ++i) {
          const vertex_t p = queue[i];
          count += child_start[p + 1] - child_start[p];
        }
        sums[t] = count;
#pragma omp barrier
#pragma omp single
        {
          std::int64_t offset = 0;
          for (std::int64_t i = 0; i < team; ++i) {
            const std::int64_t c = sums[i];
            sums[i] = offset;
            offset += c;
          }
          sums[team] = offset;
        }
        std::int64_t position = here.end + sums[t];
        for (std::int64_t i = first; i < last; ++i) {
          const vertex_t p = queue[i];
          for (vertex_t c = child_start[p]; c < child_start[p + 1]; ++c) {
            const vertex_t v = children[c];
            (void)relax(p, v);
            queue[position++] = v;
          }
        }
#pragma omp barrier
#pragma omp single
        next = path_level{here.end, here.end + sums[team],
                          __atomic_load_n(&missing, __ATOMIC_RELAXED)};
      }
    }
  }
  if (missing) {
    // Report the vertex the sequential traversal finds first (an error path: run it again).
    return mosp_path_costs(out, parent, source, k, costs, ws);
  }
  return -1;
}

#else  // !DYNG_HAS_OPENMP

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t mosp_path_costs_openmp(const resources& /*res*/,
                                const csr_view<vertex_t, edge_t, weight_t>& out,
                                const vertex_t* parent, vertex_t source, int k, std::int64_t* costs,
                                mosp_workspace<vertex_t, edge_t, weight_t>& ws) {
  return mosp_path_costs(out, parent, source, k, costs, ws);
}

template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_openmp(
    const resources& /*res*/, const mosp_combine_input<vertex_t>& /*in*/,
    mosp_workspace<vertex_t, edge_t, weight_t>& /*ws*/) {
  throw not_supported_error("dyng: mosp: the openmp backend is not built");
}

#endif  // DYNG_HAS_OPENMP

#define DYNG_INSTANTIATE_MOSP_OPENMP(V, E, W)                                                      \
  template mosp_combined<V, E, W> mosp_combine_openmp<V, E, W>(                                    \
      const resources&, const mosp_combine_input<V>&, mosp_workspace<V, E, W>&);                   \
  template V mosp_path_costs_openmp<V, E, W>(const resources&, const csr_view<V, E, W>&, const V*, \
                                             V, int, std::int64_t*, mosp_workspace<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_OPENMP)
#undef DYNG_INSTANTIATE_MOSP_OPENMP

}  // namespace dyng::detail
