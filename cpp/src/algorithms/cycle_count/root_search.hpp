// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/openmp/openmp_johnson.cpp (count_root) and
// src/sequential/johnson.cpp (JohnsonSearch::circuit_bounded)
/**
 * @file root_search.hpp
 * @brief The path search from one root of the static counters: every simple cycle whose smallest
 *        vertex is the root, by length.
 *
 * The OpenMP counter's count_root() and the sequential Johnson's length-bounded circuit_bounded()
 * of the original are the same search: from the root, extend paths through vertices greater than
 * the root that are not on the path, count a cycle whenever an edge closes back to the root (paths
 * of at least two vertices), and stop extending at the length cap. A vertex is marked only while
 * it is on the path. The original recurses once per path vertex; this port keeps the path in an
 * explicit stack (same order of exploration, same counts), so a long path costs memory, not the
 * thread's stack.
 *
 * Without a length bound this is a plain simple-path enumeration (no blocked lists): its cost
 * grows with the number of simple paths over vertices above the root, even when there are few
 * cycles. The sequential backend uses Johnson's blocked lists in that case (static_sequential.cpp).
 */
#pragma once

#include "algorithms/cycle_count/problem.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

/**
 * @brief The reusable scratch of root searches (one per thread).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct cycle_root_scratch {
  std::vector<unsigned char> on_path;               ///< n entries, all zero between roots
  std::vector<cycle_search_frame<vertex_t>> stack;  ///< the explicit path
};

/**
 * @brief Count every simple cycle of length <= max_length whose smallest vertex is `root`.
 *
 * Each cycle of length len adds one to `histogram[len]` (cycle_count_record); the histogram grows
 * on demand, up to max_length + 1 entries.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph      The graph.
 * @param[in]     root       The root.
 * @param[in]     max_length Longest counted length (>= 2).
 * @param[in,out] histogram  The thread's histogram.
 * @param[in,out] scratch    The thread's scratch.
 * @throws std::bad_alloc if the scratch or the histogram cannot grow (the scratch is then dirty
 *         and must be discarded).
 */
template <typename vertex_t, typename edge_t>
void cycle_count_search_root(const cycle_graph<vertex_t, edge_t>& graph, const vertex_t root,
                             const std::size_t max_length, std::vector<std::uint64_t>& histogram,
                             cycle_root_scratch<vertex_t>& scratch) {
  // The search clears every mark it sets, so the marks are all-zero between roots: size them once
  // instead of clearing all V entries per root, which made the whole count O(V^2).
  if (scratch.on_path.size() != graph.vertex_count) {
    scratch.on_path.assign(graph.vertex_count, 0);
  }
  if (scratch.stack.empty()) {
    cycle_count_grow_stack(scratch.stack);
  }
  unsigned char* const on_path = scratch.on_path.data();
  const edge_t* const offsets = graph.offsets;
  const vertex_t* const neighbors = graph.neighbors;
  cycle_search_frame<vertex_t>* frames = scratch.stack.data();
  std::size_t capacity = scratch.stack.size();
  std::uint64_t* counts = histogram.data();
  std::size_t count_size = histogram.size();

  on_path[static_cast<std::size_t>(root)] = 1;
  frames[0].vertex = root;
  frames[0].next = static_cast<std::size_t>(offsets[root]);
  frames[0].end = static_cast<std::size_t>(offsets[root + 1]);
  std::size_t depth = 1;  // vertices on the path, the root included

  while (depth > 0) {
    cycle_search_frame<vertex_t>& top = frames[depth - 1];
    if (top.next == top.end) {
      on_path[static_cast<std::size_t>(top.vertex)] = 0;
      --depth;
      continue;
    }
    const vertex_t next = neighbors[top.next];
    ++top.next;

    if (next == root && depth >= 2) {
      if (depth >= count_size) {
        cycle_count_grow(histogram, depth, max_length);
        counts = histogram.data();
        count_size = histogram.size();
      }
      cycle_count_record(counts, depth);
      continue;
    }

    if (next <= root || on_path[static_cast<std::size_t>(next)] != 0) {
      continue;
    }

    if (depth >= max_length) {
      continue;
    }

    if (depth == capacity) {
      cycle_count_grow_stack(scratch.stack);
      frames = scratch.stack.data();
      capacity = scratch.stack.size();
    }
    on_path[static_cast<std::size_t>(next)] = 1;
    frames[depth].vertex = next;
    frames[depth].next = static_cast<std::size_t>(offsets[next]);
    frames[depth].end = static_cast<std::size_t>(offsets[next + 1]);
    ++depth;
  }
}

}  // namespace dyng::detail
