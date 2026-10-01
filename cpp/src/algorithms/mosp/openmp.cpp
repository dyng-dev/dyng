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
#include "core/resources_access.hpp"
#include "graph/instantiate.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

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
  const int K = in.num_objectives;

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
    for (int k = 0; k < K; ++k) {
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
    for (int k = 0; k < K; ++k) {
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

#else  // !DYNG_HAS_OPENMP

template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_openmp(
    const resources& /*res*/, const mosp_combine_input<vertex_t>& /*in*/,
    mosp_workspace<vertex_t, edge_t, weight_t>& /*ws*/) {
  throw not_supported_error("dyng: mosp: the openmp backend is not built");
}

#endif  // DYNG_HAS_OPENMP

#define DYNG_INSTANTIATE_MOSP_OPENMP(V, E, W)                   \
  template mosp_combined<V, E, W> mosp_combine_openmp<V, E, W>( \
      const resources&, const mosp_combine_input<V>&, mosp_workspace<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_OPENMP)
#undef DYNG_INSTANTIATE_MOSP_OPENMP

}  // namespace dyng::detail
