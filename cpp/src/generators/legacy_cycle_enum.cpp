// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/batch_generator.cpp (generate_batch)
/**
 * @file legacy_cycle_enum.cpp
 * @brief generators::legacy::cycle_enum_batch(): CycleEnumeration-GPU's seeded batch generator,
 *        bit-exact.
 *
 * Mechanical changes only: names, templates on the index types, invalid_argument_error instead of
 * std::invalid_argument, a csr_view instead of the GraphView (whose rows are sorted by neighbour,
 * as the view of a cycle_enum_compatible() graph), the draws of std::uniform_int_distribution
 * <size_t> and the order of std::shuffle through detail::legacy_uniform_int64 and
 * detail::legacy_shuffle (libstdc++'s algorithms, so the stream does not depend on the standard
 * library), a set of (u, v) pairs instead of the 64-bit key (u << 32) | v (the same membership for
 * 32-bit ids, exact for 64-bit ids), and weight 1 in every column of a weighted graph.
 */
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/rng.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/legacy.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dyng::generators::legacy {

namespace {

/// Exact membership of (u, v) pairs (the original's unordered_set of edge keys).
template <typename vertex_t>
struct pair_hash {
  std::size_t operator()(const std::pair<vertex_t, vertex_t>& p) const noexcept {
    const auto u = static_cast<std::uint64_t>(p.first);
    const auto v = static_cast<std::uint64_t>(p.second);
    return std::hash<std::uint64_t>{}((u << 32U) ^ (v * 0x9E3779B97F4A7C15ULL));
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
edge_batch<vertex_t, weight_t> cycle_enum_batch(const csr_view<vertex_t, edge_t, weight_t>& graph,
                                                const cycle_enum_batch_options& opt) try {
  DYNG_EXPECTS(opt.num_deletions >= 0 && opt.num_insertions >= 0,
               "cycle_enum_batch: the counts must be >= 0, got ", opt.num_deletions,
               " deletions and ", opt.num_insertions, " insertions");
  DYNG_EXPECTS(graph.row_ptr.empty() || is_host_accessible(graph.row_ptr.space()),
               "cycle_enum_batch: the graph must be in host-accessible memory");
  DYNG_EXPECTS(graph.col_ind.empty() || is_host_accessible(graph.col_ind.space()),
               "cycle_enum_batch: the graph must be in host-accessible memory");
  const int num_weights = graph.num_weights;
  edge_batch<vertex_t, weight_t> batch(num_weights);
  if (opt.num_deletions == 0 && opt.num_insertions == 0) {
    return batch;
  }

  const auto vertex_count = static_cast<std::uint64_t>(graph.num_vertices());
  DYNG_EXPECTS(vertex_count >= 2,
               "cycle_enum_batch: batch generation requires at least two "
               "vertices");
  const auto row_begin = [&](std::uint64_t u) {
    return static_cast<std::size_t>(graph.row_ptr[static_cast<std::size_t>(u)]);
  };
  const auto row_end = [&](std::uint64_t u) {
    return static_cast<std::size_t>(graph.row_ptr[static_cast<std::size_t>(u) + 1]);
  };

  using edge_pair = std::pair<vertex_t, vertex_t>;
  std::vector<edge_pair> existing_edges;
  existing_edges.reserve(graph.col_ind.size());
  for (std::uint64_t source = 0; source < vertex_count; ++source) {
    for (std::size_t offset = row_begin(source); offset < row_end(source); ++offset) {
      const vertex_t target = graph.col_ind[offset];
      DYNG_EXPECTS(offset == row_begin(source) || graph.col_ind[offset - 1] < target,
                   "cycle_enum_batch: row ", source,
                   " is not sorted without repeats (build the graph with "
                   "graph_properties::cycle_enum_compatible())");
      existing_edges.emplace_back(static_cast<vertex_t>(source), target);
    }
  }
  // Rows are sorted by neighbour, so an edge lookup is a binary search.
  const auto exists = [&](vertex_t source, vertex_t target) {
    const auto begin = graph.col_ind.begin() +
                       static_cast<std::ptrdiff_t>(row_begin(static_cast<std::uint64_t>(source)));
    const auto end = graph.col_ind.begin() +
                     static_cast<std::ptrdiff_t>(row_end(static_cast<std::uint64_t>(source)));
    return std::binary_search(begin, end, target);
  };

  std::mt19937_64 rng(opt.seed);

  std::uint64_t window_lo = 0;
  std::uint64_t window_hi = vertex_count;
  if (opt.locality_window >= 0 && static_cast<std::uint64_t>(opt.locality_window) < vertex_count) {
    const std::uint64_t window =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(opt.locality_window), 2);
    if (window < vertex_count) {
      const std::uint64_t start = detail::legacy_uniform_int64(rng, 0, vertex_count - window);
      window_lo = start;
      window_hi = start + window;
    }
  }
  const auto in_window = [&](vertex_t vertex) {
    const auto v = static_cast<std::uint64_t>(vertex);
    return v >= window_lo && v < window_hi;
  };

  std::vector<edge_pair> deletions;
  if (opt.num_deletions > 0) {
    std::vector<edge_pair> candidates;
    for (const edge_pair& edge : existing_edges) {
      if (in_window(edge.first) && in_window(edge.second)) {
        candidates.push_back(edge);
      }
    }
    DYNG_EXPECTS(candidates.size() >= static_cast<std::size_t>(opt.num_deletions),
                 "cycle_enum_batch: not enough existing edges in the locality window for the "
                 "requested deletions (",
                 candidates.size(), " < ", opt.num_deletions, ")");
    detail::legacy_shuffle(candidates, rng);
    deletions.assign(candidates.begin(),
                     candidates.begin() + static_cast<std::ptrdiff_t>(opt.num_deletions));
  }

  std::vector<edge_pair> insertions;
  if (opt.num_insertions > 0) {
    const std::uint64_t window = window_hi - window_lo;
    std::unordered_set<edge_pair, pair_hash<vertex_t>> chosen;
    const auto wanted = static_cast<std::uint64_t>(opt.num_insertions);
    const std::uint64_t attempt_cap = 1000ULL * (wanted + 1) + 100ULL * window;
    std::uint64_t attempts = 0;
    while (insertions.size() < wanted) {
      DYNG_EXPECTS(++attempts <= attempt_cap,
                   "cycle_enum_batch: could not sample enough non-edges in the locality window "
                   "for the requested insertions");
      const auto source =
          static_cast<vertex_t>(detail::legacy_uniform_int64(rng, window_lo, window_hi - 1));
      const auto target =
          static_cast<vertex_t>(detail::legacy_uniform_int64(rng, window_lo, window_hi - 1));
      if (source == target) {
        continue;
      }
      const edge_pair key{source, target};
      if (exists(source, target) || chosen.count(key) != 0) {
        continue;
      }
      chosen.insert(key);
      insertions.push_back(key);
    }
  }

  // normalize(): each list sorted by (source, target) and duplicate-free.
  for (std::vector<edge_pair>* list : {&deletions, &insertions}) {
    std::sort(list->begin(), list->end());
    list->erase(std::unique(list->begin(), list->end()), list->end());
  }
  batch.reserve(insertions.size(), deletions.size());
  for (const edge_pair& d : deletions) {
    batch.delete_edge(d.first, d.second);
  }
  const std::vector<weight_t> weights(static_cast<std::size_t>(num_weights), [] {
    if constexpr (is_unweighted_v<weight_t>) {
      return weight_t{};
    } else {
      return weight_t{1};
    }
  }());
  for (const edge_pair& i : insertions) {
    batch.insert_edge(i.first, i.second, host_view(weights));
  }
  return batch;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::cycle_enum_batch (", opt.num_deletions,
                                  " deletions, ", opt.num_insertions, " insertions)")

#define DYNG_INSTANTIATE_CYCLE_ENUM_BATCH(V, E, W)                              \
  template edge_batch<V, W> cycle_enum_batch<V, E, W>(const csr_view<V, E, W>&, \
                                                      const cycle_enum_batch_options&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_CYCLE_ENUM_BATCH)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_CYCLE_ENUM_BATCH)
#undef DYNG_INSTANTIATE_CYCLE_ENUM_BATCH

}  // namespace dyng::generators::legacy
