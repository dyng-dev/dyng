// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/combinedGraphCpu.cpp (combinedEdge, combinedGraphSospCpu
// Step 2, mospPathCosts)
/**
 * @file sequential.cpp
 * @brief The host pieces of mosp's finalize step: the combined graph on the sequential backend
 *        (MOSP-OpenMP's combinedGraphSospCpu, Step 2, without the threads), the path costs of
 *        every backend (mospPathCosts) and the host workspace.
 *
 * The combined graph: edge (p, v) exists iff p is the parent of v in some tree T_i; its weight is
 * L * (K + 1) - sum_{i : Parent_i[v] == p} L / Pref_i. The in-edges of v are the distinct values
 * among Parent_0[v] .. Parent_{K-1}[v], so each vertex compares its K parents (combined_edge: the
 * first occurrence of a value carries the edge and its weight) and a count / prefix-sum / fill
 * pass builds the out-edge CSR. The order inside a row does not matter: the solve breaks ties by
 * the lowest parent id.
 *
 * Changes from the original: templates on the index types; the parents of the K trees are K
 * arrays (the sssp results) instead of one objective-major array; the in-edge CSR is built as
 * well when the sequential engine needs it (its distance-only parent recovery reads in-edges;
 * MOSP's OpenMP engine reads out-edges there); mospPathCosts reports the first vertex whose tree
 * edge is missing instead of returning false, and keeps its arrays in the pooled workspace.
 */
#include "algorithms/mosp/problem.hpp"
#include "core/budget_counters.hpp"
#include "graph/instantiate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

namespace {

/**
 * MOSP's combinedEdge(): if parent k of v is the first occurrence of its value among the K
 * parents, return true and its combined-graph weight base - sum_{j : Parent_j[v] == p} terms[j].
 */
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
void mosp_workspace<vertex_t, edge_t, weight_t>::reserve(std::size_t n, int k) {
  const std::size_t edges = n * static_cast<std::size_t>(std::max(k, 1));
  if (row_ptr.capacity() >= n + 1 && col_ind.capacity() >= edges && queue.capacity() >= n) {
    return;
  }
  note_reservation();  // invariant I9: a reserving run (core/budget_counters.hpp)
  row_ptr.reserve(n + 1);
  cursor.reserve(n + 1);
  col_ind.reserve(edges);
  weights.reserve(edges);
  child_start.reserve(n + 1);
  children.reserve(n);
  queue.reserve(n);
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::size_t mosp_workspace<vertex_t, edge_t, weight_t>::bytes() const noexcept {
  return (row_ptr.capacity() + cursor.capacity() + in_row_ptr.capacity()) * sizeof(edge_t) +
         (col_ind.capacity() + in_col_ind.capacity() + child_start.capacity() +
          children.capacity() + queue.capacity()) *
             sizeof(vertex_t) +
         (weights.capacity() + in_weights.capacity()) * sizeof(weight_t);
}

template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_sequential(
    const mosp_combine_input<vertex_t>& in, mosp_workspace<vertex_t, edge_t, weight_t>& ws,
    bool with_in_edges) {
  const vertex_t n = in.num_vertices;
  const auto rows = static_cast<std::size_t>(n) + 1;
  // Count the out-degrees, prefix sum, fill (combinedGraphSospCpu, Step 2).
  ws.cursor.assign(rows, edge_t{0});
  std::int64_t edges = 0;
  std::int64_t weight_sum = 0;
  for (vertex_t v = 0; v < n; ++v) {
    if (v == in.source) {
      continue;
    }
    for (int k = 0; k < in.num_objectives; ++k) {
      vertex_t p = 0;
      weight_t weight = 0;
      if (combined_edge(in, v, k, p, weight)) {
        ++ws.cursor[static_cast<std::size_t>(p)];
        ++edges;
        weight_sum += weight;
      }
    }
  }
  ws.row_ptr.resize(rows);
  ws.row_ptr[0] = 0;
  for (std::size_t u = 0; u + 1 < rows; ++u) {
    ws.row_ptr[u + 1] = static_cast<edge_t>(ws.row_ptr[u] + ws.cursor[u]);
  }
  std::copy(ws.row_ptr.begin(), ws.row_ptr.end() - 1, ws.cursor.begin());
  ws.col_ind.resize(static_cast<std::size_t>(edges));
  ws.weights.resize(static_cast<std::size_t>(edges));
  for (vertex_t v = 0; v < n; ++v) {
    if (v == in.source) {
      continue;
    }
    for (int k = 0; k < in.num_objectives; ++k) {
      vertex_t p = 0;
      weight_t weight = 0;
      if (combined_edge(in, v, k, p, weight)) {
        const auto position = static_cast<std::size_t>(ws.cursor[static_cast<std::size_t>(p)]++);
        ws.col_ind[position] = v;
        ws.weights[position] = weight;
      }
    }
  }
  mosp_combined<vertex_t, edge_t, weight_t> out;
  out.view.num_vertices = n;
  out.view.out_row_ptr = ws.row_ptr.data();
  out.view.out_col_ind = ws.col_ind.data();
  out.view.out_weights = ws.weights.data();
  out.edges = edges;
  out.weight_sum = weight_sum;
  if (with_in_edges) {
    // The in-edges of v are its distinct parents: row v lists them in objective order.
    ws.in_row_ptr.assign(rows, edge_t{0});
    ws.in_col_ind.resize(static_cast<std::size_t>(edges));
    ws.in_weights.resize(static_cast<std::size_t>(edges));
    std::size_t position = 0;
    for (vertex_t v = 0; v < n; ++v) {
      if (v != in.source) {
        for (int k = 0; k < in.num_objectives; ++k) {
          vertex_t p = 0;
          weight_t weight = 0;
          if (combined_edge(in, v, k, p, weight)) {
            ws.in_col_ind[position] = p;
            ws.in_weights[position] = weight;
            ++position;
          }
        }
      }
      ws.in_row_ptr[static_cast<std::size_t>(v) + 1] = static_cast<edge_t>(position);
    }
    out.view.in_row_ptr = ws.in_row_ptr.data();
    out.view.in_col_ind = ws.in_col_ind.data();
    out.view.in_weights = ws.in_weights.data();
  }
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t mosp_path_costs(const csr_view<vertex_t, edge_t, weight_t>& out, const vertex_t* parent,
                         vertex_t source, int k, std::int64_t* costs,
                         mosp_workspace<vertex_t, edge_t, weight_t>& ws) {
  const vertex_t n = out.num_vertices();
  const auto K = static_cast<std::size_t>(k);
  std::fill(costs, costs + static_cast<std::size_t>(n) * K, sssp_infinity);
  if (n == 0) {
    return -1;
  }
  // Children lists of the tree, then a traversal from the source.
  const auto rows = static_cast<std::size_t>(n) + 1;
  ws.child_start.assign(rows, vertex_t{0});
  for (vertex_t v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      ++ws.child_start[static_cast<std::size_t>(parent[v]) + 1];
    }
  }
  for (vertex_t v = 0; v < n; ++v) {
    ws.child_start[static_cast<std::size_t>(v) + 1] += ws.child_start[static_cast<std::size_t>(v)];
  }
  // The fill cursor reuses `queue` (its entries are written again by the traversal below).
  ws.queue.assign(ws.child_start.begin(), ws.child_start.end() - 1);
  ws.children.resize(static_cast<std::size_t>(n));
  for (vertex_t v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      ws.children[static_cast<std::size_t>(ws.queue[static_cast<std::size_t>(parent[v])]++)] = v;
    }
  }
  for (std::size_t j = 0; j < K; ++j) {
    costs[static_cast<std::size_t>(source) * K + j] = 0;
  }
  const edge_t* row_ptr = out.row_ptr.data();
  const vertex_t* col_ind = out.col_ind.data();
  const weight_t* weights = out.weights.data();
  const auto m = static_cast<std::size_t>(out.col_ind.size());
  ws.queue.clear();
  ws.queue.push_back(source);
  for (std::size_t i = 0; i < ws.queue.size(); ++i) {
    const vertex_t p = ws.queue[i];
    for (vertex_t c = ws.child_start[static_cast<std::size_t>(p)];
         c < ws.child_start[static_cast<std::size_t>(p) + 1]; ++c) {
      const vertex_t v = ws.children[static_cast<std::size_t>(c)];
      edge_t edge = -1;
      for (edge_t e = row_ptr[p]; e < row_ptr[p + 1]; ++e) {
        if (col_ind[e] == v) {
          edge = e;
          break;
        }
      }
      if (edge < 0) {
        return v;
      }
      for (std::size_t j = 0; j < K; ++j) {
        costs[static_cast<std::size_t>(v) * K + j] =
            costs[static_cast<std::size_t>(p) * K + j] +
            static_cast<std::int64_t>(weights[j * m + static_cast<std::size_t>(edge)]);
      }
      ws.queue.push_back(v);
    }
  }
  return -1;
}

#define DYNG_INSTANTIATE_MOSP_HOST(V, E, W)                                                      \
  template struct mosp_workspace<V, E, W>;                                                       \
  template mosp_combined<V, E, W> mosp_combine_sequential<V, E, W>(                              \
      const mosp_combine_input<V>&, mosp_workspace<V, E, W>&, bool);                             \
  template V mosp_path_costs<V, E, W>(const csr_view<V, E, W>&, const V*, V, int, std::int64_t*, \
                                      mosp_workspace<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_HOST)
#undef DYNG_INSTANTIATE_MOSP_HOST

}  // namespace dyng::detail
