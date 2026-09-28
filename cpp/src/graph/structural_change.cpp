// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/directed_graph.cpp (prepare_batch,
// has_edge) and src/dynamic/edge_change.cpp (sort_and_dedup)
/**
 * @file structural_change.cpp
 * @brief compute_structural_change(): CycleEnumeration-GPU's prepare_batch() on G_t, for every
 *        batch_semantics (see structural_change.hpp).
 */
#include "graph/structural_change.hpp"

#include "graph/instantiate.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dyng::detail {

namespace {

/// sort_and_dedup(): sorted by (source, target), repeats removed.
template <typename vertex_t>
void sort_and_dedup(std::vector<edge_change<vertex_t>>& changes) {
  std::sort(changes.begin(), changes.end());
  changes.erase(std::unique(changes.begin(), changes.end()), changes.end());
}

/// has_edge(): binary search in the sorted row of `source` (false for a vertex >= n).
template <typename vertex_t, typename edge_t, typename weight_t>
bool has_edge(const csr_view<vertex_t, edge_t, weight_t>& g, vertex_t source, vertex_t target) {
  if (static_cast<std::int64_t>(source) >= static_cast<std::int64_t>(g.num_vertices())) {
    return false;
  }
  const vertex_t* begin = g.col_ind.data() + static_cast<std::ptrdiff_t>(g.row_ptr[source]);
  const vertex_t* end = g.col_ind.data() + static_cast<std::ptrdiff_t>(g.row_ptr[source + 1]);
  return std::binary_search(begin, end, target);
}

/// Append the change (u, v) (and (v, u) for an undirected graph); self-loops and negative ids are
/// skipped.
template <typename vertex_t>
void add_change(std::vector<edge_change<vertex_t>>& list, vertex_t u, vertex_t v, bool directed) {
  if (u < 0 || v < 0 || u == v) {
    return;
  }
  list.push_back({u, v});
  if (!directed) {
    list.push_back({v, u});
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_structural_change(const csr_view<vertex_t, edge_t, weight_t>& g,
                               const edge_batch_view<vertex_t, weight_t>& batch,
                               const graph_properties& props, structural_change<vertex_t>& out) {
  DYNG_EXPECTS(props.order == row_order::sorted && props.parallel_edges == multi_edges::forbid,
               "the structural change of a batch needs sorted rows without parallel edges");
  DYNG_EXPECTS(batch.delete_src.size() == batch.delete_dst.size() &&
                   batch.insert_src.size() == batch.insert_dst.size(),
               "the batch has ", batch.delete_src.size(), " deletion sources for ",
               batch.delete_dst.size(), " destinations and ", batch.insert_src.size(),
               " insertion sources for ", batch.insert_dst.size(), " destinations");
  DYNG_EXPECTS((batch.delete_src.empty() || is_host_accessible(batch.delete_src.space())) &&
                   (batch.delete_dst.empty() || is_host_accessible(batch.delete_dst.space())) &&
                   (batch.insert_src.empty() || is_host_accessible(batch.insert_src.space())) &&
                   (batch.insert_dst.empty() || is_host_accessible(batch.insert_dst.space())),
               "the batch must be in host memory");
  const bool directed = props.directed;
  const batch_semantics& semantics = props.semantics;
  const auto n = static_cast<std::int64_t>(g.num_vertices());

  // The requested deletions, sorted and deduplicated; the kept ones exist in G_t.
  out.requested.clear();
  out.requested.reserve(directed ? batch.delete_src.size() : 2 * batch.delete_src.size());
  for (std::size_t j = 0; j < batch.delete_src.size(); ++j) {
    add_change(out.requested, batch.delete_src[j], batch.delete_dst[j], directed);
  }
  sort_and_dedup(out.requested);
  out.deletions.clear();
  out.deletions.reserve(out.requested.size());
  for (const edge_change<vertex_t>& d : out.requested) {
    if (has_edge(g, d.source, d.target)) {
      out.deletions.push_back(d);
    }
  }

  // The requested insertions, sorted and deduplicated; the kept ones change the edge set.
  out.insertions.clear();
  out.insertions.reserve(directed ? batch.insert_src.size() : 2 * batch.insert_src.size());
  for (std::size_t i = 0; i < batch.insert_src.size(); ++i) {
    const vertex_t u = batch.insert_src[i];
    const vertex_t v = batch.insert_dst[i];
    if (!semantics.allow_vertex_growth &&
        (static_cast<std::int64_t>(u) >= n || static_cast<std::int64_t>(v) >= n)) {
      continue;  // the commit rejects the batch
    }
    add_change(out.insertions, u, v, directed);
  }
  sort_and_dedup(out.insertions);
  std::size_t kept = 0;
  for (const edge_change<vertex_t>& c : out.insertions) {
    bool keep = false;
    if (has_edge(g, c.source, c.target)) {
      // An existing edge: a no-op, unless the batch deletes it first and adds it back.
      keep = semantics.deletions_first &&
             std::binary_search(out.deletions.begin(), out.deletions.end(), c);
    } else {
      // A new edge, unless it is inserted and then deleted again (insertions first).
      keep = semantics.deletions_first ||
             !std::binary_search(out.requested.begin(), out.requested.end(), c);
    }
    if (keep) {
      out.insertions[kept++] = c;
    }
  }
  out.insertions.resize(kept);
}

#define DYNG_INSTANTIATE_STRUCTURAL_CHANGE(V, E, W)                                    \
  template void compute_structural_change<V, E, W>(                                    \
      const csr_view<V, E, W>&, const edge_batch_view<V, W>&, const graph_properties&, \
      structural_change<V>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_STRUCTURAL_CHANGE)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_STRUCTURAL_CHANGE)
#undef DYNG_INSTANTIATE_STRUCTURAL_CHANGE

}  // namespace dyng::detail
