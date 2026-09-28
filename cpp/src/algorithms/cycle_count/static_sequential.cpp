// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/sequential/johnson.cpp (JohnsonSearch,
// count_simple_cycles_johnson)
/**
 * @file static_sequential.cpp
 * @brief compute() on the sequential backend: CycleEnumeration-GPU's sequential Johnson search.
 *
 * Straight port of JohnsonSearch: each directed cycle is counted once, from its smallest vertex
 * as the root. With a length bound the search blocks a vertex only while it is on the current
 * path (Johnson's blocked lists are unsound together with a depth cutoff: a vertex whose search
 * failed only because of the cutoff would stay blocked and shorter cycles through it would be
 * missed); without a bound it is Johnson's algorithm with blocked lists, resetting only what a
 * root's search touched. The mechanical changes: the templates on the index types, the graph as a
 * cycle_graph, and a dense counts array instead of the CycleHistogram map.
 */
#include "algorithms/cycle_count/problem.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

namespace {

template <typename vertex_t, typename edge_t>
class johnson_search {
 public:
  johnson_search(const cycle_graph<vertex_t, edge_t>& graph, std::int64_t max_cycle_length,
                 std::vector<std::uint64_t>& counts)
      : graph_(graph),
        bounded_(max_cycle_length >= 0),
        max_cycle_length_(bounded_ ? static_cast<std::size_t>(max_cycle_length) : 0),
        counts_(counts),
        blocked_(graph.vertex_count, 0),
        touched_mark_(graph.vertex_count, 0) {
    if (!bounded_) {
      blocked_list_.resize(graph.vertex_count);
    }
    path_.reserve(graph.vertex_count);
  }

  void run() {
    for (std::size_t root = 0; root < graph_.vertex_count; ++root) {
      root_ = static_cast<vertex_t>(root);
      path_.clear();
      if (bounded_) {
        // circuit_bounded restores blocked_ on every backtrack and never uses the blocked
        // lists, so no per-root reset is needed.
        circuit_bounded(root_);
      } else {
        circuit(root_);
        // Reset only what this root's search touched instead of all V entries, which made the
        // whole run O(V^2).
        for (const vertex_t vertex : touched_) {
          blocked_[static_cast<std::size_t>(vertex)] = 0;
          blocked_list_[static_cast<std::size_t>(vertex)].clear();
          touched_mark_[static_cast<std::size_t>(vertex)] = 0;
        }
        touched_.clear();
      }
    }
  }

 private:
  // Length-bounded search. Johnson's blocked-list unblocking is only sound for unbounded search:
  // when a vertex's search fails purely because the depth limit was hit, the blocked-list
  // mechanism would keep it blocked and miss shorter cycles through it. With a length bound we
  // therefore fall back to plain path-membership blocking, marking a vertex blocked only while it
  // is on the current path and clearing it on every backtrack.
  bool circuit_bounded(const vertex_t vertex) {
    path_.push_back(vertex);
    blocked_[static_cast<std::size_t>(vertex)] = 1;

    bool found_cycle = false;
    const auto begin = static_cast<std::size_t>(graph_.offsets[vertex]);
    const auto end = static_cast<std::size_t>(graph_.offsets[vertex + 1]);
    for (std::size_t offset = begin; offset < end; ++offset) {
      const vertex_t next = graph_.neighbors[offset];

      if (next == root_ && path_.size() >= 2) {
        cycle_count_record(counts_, path_.size());
        found_cycle = true;
        continue;
      }

      if (next <= root_ || blocked_[static_cast<std::size_t>(next)] != 0) {
        continue;
      }

      if (path_.size() >= max_cycle_length_) {
        continue;
      }

      if (circuit_bounded(next)) {
        found_cycle = true;
      }
    }

    blocked_[static_cast<std::size_t>(vertex)] = 0;
    path_.pop_back();
    return found_cycle;
  }

  bool circuit(const vertex_t vertex) {
    bool found_cycle = false;
    path_.push_back(vertex);
    blocked_[static_cast<std::size_t>(vertex)] = 1;
    touch(vertex);

    const auto begin = static_cast<std::size_t>(graph_.offsets[vertex]);
    const auto end = static_cast<std::size_t>(graph_.offsets[vertex + 1]);
    for (std::size_t offset = begin; offset < end; ++offset) {
      const vertex_t next = graph_.neighbors[offset];

      if (next == root_ && path_.size() >= 2) {
        cycle_count_record(counts_, path_.size());
        found_cycle = true;
        continue;
      }

      if (next <= root_ || blocked_[static_cast<std::size_t>(next)] != 0) {
        continue;
      }

      if (circuit(next)) {
        found_cycle = true;
      }
    }

    if (found_cycle) {
      unblock(vertex);
    } else {
      add_to_blocked_lists(vertex);
    }

    path_.pop_back();
    return found_cycle;
  }

  // Record a vertex whose blocked state or blocked list must be reset. Each vertex is listed
  // once, so touched_ holds at most V entries however many times the search enters the vertex.
  void touch(const vertex_t vertex) {
    if (touched_mark_[static_cast<std::size_t>(vertex)] == 0) {
      touched_mark_[static_cast<std::size_t>(vertex)] = 1;
      touched_.push_back(vertex);
    }
  }

  void unblock(const vertex_t vertex) {
    blocked_[static_cast<std::size_t>(vertex)] = 0;
    std::vector<vertex_t>& dependents = blocked_list_[static_cast<std::size_t>(vertex)];
    while (!dependents.empty()) {
      const vertex_t dependent = dependents.back();
      dependents.pop_back();
      if (blocked_[static_cast<std::size_t>(dependent)] != 0) {
        unblock(dependent);
      }
    }
  }

  void add_to_blocked_lists(const vertex_t vertex) {
    const auto begin = static_cast<std::size_t>(graph_.offsets[vertex]);
    const auto end = static_cast<std::size_t>(graph_.offsets[vertex + 1]);
    for (std::size_t offset = begin; offset < end; ++offset) {
      const vertex_t next = graph_.neighbors[offset];
      if (next <= root_) {
        continue;
      }

      std::vector<vertex_t>& dependents = blocked_list_[static_cast<std::size_t>(next)];
      if (std::find(dependents.begin(), dependents.end(), vertex) == dependents.end()) {
        dependents.push_back(vertex);
        touch(next);
      }
    }
  }

  const cycle_graph<vertex_t, edge_t>& graph_;
  bool bounded_;
  std::size_t max_cycle_length_;
  std::vector<std::uint64_t>& counts_;
  std::vector<unsigned char> blocked_;
  std::vector<std::vector<vertex_t>> blocked_list_;
  std::vector<vertex_t> touched_;
  std::vector<unsigned char> touched_mark_;
  std::vector<vertex_t> path_;
  vertex_t root_ = 0;
};

}  // namespace

template <typename vertex_t, typename edge_t>
void cycle_count_sequential_compute(const cycle_graph<vertex_t, edge_t>& graph,
                                    std::int64_t max_length, std::vector<std::uint64_t>& counts) {
  johnson_search<vertex_t, edge_t>(graph, max_length, counts).run();
}

template void cycle_count_sequential_compute<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, std::int64_t, std::vector<std::uint64_t>&);
template void cycle_count_sequential_compute<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, std::int64_t, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
