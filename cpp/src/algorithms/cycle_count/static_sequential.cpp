// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/sequential/johnson.cpp (JohnsonSearch,
// count_simple_cycles_johnson)
/**
 * @file static_sequential.cpp
 * @brief compute() on the sequential backend: CycleEnumeration-GPU's sequential Johnson search.
 *
 * Port of JohnsonSearch: each directed cycle is counted once, from its smallest vertex as the
 * root. With a length bound the search blocks a vertex only while it is on the current path
 * (Johnson's blocked lists are unsound together with a depth cutoff: a vertex whose search failed
 * only because of the cutoff would stay blocked and shorter cycles through it would be missed);
 * that search is the one of the OpenMP counter (root_search.hpp). Without a bound it is Johnson's
 * algorithm with blocked lists, resetting only what a root's search touched. The mechanical
 * changes: the templates on the index types, the graph as a cycle_graph, a dense counts array
 * instead of the CycleHistogram map, and explicit stacks for the recursions of circuit() and
 * unblock() (the original recurses once per path vertex, which overflows the thread's stack on
 * long paths; the order of exploration and the counts are the same).
 */
#include "algorithms/cycle_count/problem.hpp"
#include "algorithms/cycle_count/root_search.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

namespace {

/// One level of circuit(): the frame of the search plus whether a cycle was found below it.
template <typename vertex_t>
struct johnson_frame {
  vertex_t vertex{};
  std::size_t next = 0;
  std::size_t end = 0;
  bool found_cycle = false;
};

/// Johnson's algorithm with blocked lists (the unbounded search of JohnsonSearch).
template <typename vertex_t, typename edge_t>
class johnson_search {
 public:
  johnson_search(const cycle_graph<vertex_t, edge_t>& graph, std::vector<std::uint64_t>& counts)
      : graph_(graph),
        counts_(counts),
        blocked_(graph.vertex_count, 0),
        blocked_list_(graph.vertex_count),
        touched_mark_(graph.vertex_count, 0) {}

  void run() {
    for (std::size_t root = 0; root < graph_.vertex_count; ++root) {
      root_ = static_cast<vertex_t>(root);
      circuit();
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

 private:
  // circuit(root) of the original with the recursion on an explicit stack: entering a vertex
  // pushes a frame (path_.push_back, blocked, touch); leaving it unblocks the vertex if a cycle
  // was found below it, else adds it to the blocked lists of its successors, and passes
  // found_cycle up to the parent frame.
  void circuit() {
    enter(root_);
    while (!stack_.empty()) {
      johnson_frame<vertex_t>& top = stack_.back();
      if (top.next == top.end) {
        const vertex_t vertex = top.vertex;
        const bool found_cycle = top.found_cycle;
        if (found_cycle) {
          unblock(vertex);
        } else {
          add_to_blocked_lists(vertex);
        }
        stack_.pop_back();
        if (found_cycle && !stack_.empty()) {
          stack_.back().found_cycle = true;
        }
        continue;
      }
      const vertex_t next = graph_.neighbors[top.next];
      ++top.next;

      if (next == root_ && stack_.size() >= 2) {
        cycle_count_record(counts_.data(), stack_.size());
        top.found_cycle = true;
        continue;
      }

      if (next <= root_ || blocked_[static_cast<std::size_t>(next)] != 0) {
        continue;
      }

      enter(next);  // may reallocate the stack: `top` is not used after this
    }
  }

  void enter(const vertex_t vertex) {
    johnson_frame<vertex_t> frame;
    frame.vertex = vertex;
    frame.next = static_cast<std::size_t>(graph_.offsets[vertex]);
    frame.end = static_cast<std::size_t>(graph_.offsets[vertex + 1]);
    stack_.push_back(frame);
    blocked_[static_cast<std::size_t>(vertex)] = 1;
    touch(vertex);
  }

  // Record a vertex whose blocked state or blocked list must be reset. Each vertex is listed
  // once, so touched_ holds at most V entries however many times the search enters the vertex.
  void touch(const vertex_t vertex) {
    if (touched_mark_[static_cast<std::size_t>(vertex)] == 0) {
      touched_mark_[static_cast<std::size_t>(vertex)] = 1;
      touched_.push_back(vertex);
    }
  }

  // unblock(vertex) of the original with the recursion on a work list: every vertex reachable from
  // `vertex` through the blocked lists of blocked vertices is unblocked and its list emptied (the
  // same final state as the recursion, whatever the order).
  void unblock(const vertex_t vertex) {
    blocked_[static_cast<std::size_t>(vertex)] = 0;
    unblock_work_.push_back(vertex);
    while (!unblock_work_.empty()) {
      const vertex_t current = unblock_work_.back();
      unblock_work_.pop_back();
      std::vector<vertex_t>& dependents = blocked_list_[static_cast<std::size_t>(current)];
      while (!dependents.empty()) {
        const vertex_t dependent = dependents.back();
        dependents.pop_back();
        if (blocked_[static_cast<std::size_t>(dependent)] != 0) {
          blocked_[static_cast<std::size_t>(dependent)] = 0;
          unblock_work_.push_back(dependent);
        }
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
  std::vector<std::uint64_t>& counts_;
  std::vector<unsigned char> blocked_;
  std::vector<std::vector<vertex_t>> blocked_list_;
  std::vector<vertex_t> touched_;
  std::vector<unsigned char> touched_mark_;
  std::vector<johnson_frame<vertex_t>> stack_;
  std::vector<vertex_t> unblock_work_;
  vertex_t root_ = 0;
};

}  // namespace

template <typename vertex_t, typename edge_t>
void cycle_count_sequential_compute(const cycle_graph<vertex_t, edge_t>& graph,
                                    std::int64_t max_length, std::vector<std::uint64_t>& counts) {
  if (max_length < 0) {
    johnson_search<vertex_t, edge_t>(graph, counts).run();
    return;
  }
  // circuit_bounded restores the marks on every backtrack and never uses the blocked lists, so no
  // per-root reset is needed. counts has min(max_length, max(n, 2)) + 1 entries: its last index is
  // the cap (no simple cycle is longer than n).
  cycle_root_scratch<vertex_t> scratch;
  const std::size_t cap = counts.size() - 1;
  for (std::size_t root = 0; root < graph.vertex_count; ++root) {
    cycle_count_search_root(graph, static_cast<vertex_t>(root), cap, counts, scratch);
  }
}

template void cycle_count_sequential_compute<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, std::int64_t, std::vector<std::uint64_t>&);
template void cycle_count_sequential_compute<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, std::int64_t, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
