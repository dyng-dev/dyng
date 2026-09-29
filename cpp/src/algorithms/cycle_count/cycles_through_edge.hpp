// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/cycles_through_edge.cpp
// (count_cycles_through_edge, search)
/**
 * @file cycles_through_edge.hpp
 * @brief Edge-anchored simple-cycle enumeration with edge-id ownership: the count of one change
 *        edge in the count(-) and count(+) hooks of update().
 *
 * Port of count_cycles_through_edge(): the mechanical changes are the templates on the index
 * types, the graph as a cycle_graph, and the scratch passed in by the caller (the original keeps a
 * `static thread_local` visited buffer; dynG keeps one per thread in the pooled workspace, no
 * globals). Two changes of form keep the counts and the order of the search: the recursion of
 * search() is an explicit stack (the original recurses once per path vertex, which overflows the
 * thread's stack on long cycles of an unbounded update), and each cycle is added to the thread's
 * phase counters directly (the original fills and sums a per-edge array of max_length + 1 entries
 * for every change edge, O(max_length) per edge even when the edge closes no cycle).
 */
#pragma once

#include "algorithms/cycle_count/problem.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

/**
 * @brief Count the simple cycles through the directed edge (source -> target) that it owns,
 *        bucketed by length (CycleEnumeration-GPU's count_cycles_through_edge).
 *
 * The cycle is source -> target -> ... -> source. It is owned by this edge iff it contains no
 * same-phase changed edge with a smaller id, which the search enforces by refusing to cross such
 * edges. Each found cycle of length len (2 <= len <= max_length) adds one to
 * `scratch.partial[len]`, which grows on demand.
 *
 * The depth-first search looks for simple paths from the anchored target back to the anchored
 * source; a path of `depth` vertices from the target that closes at the source is a cycle of
 * length depth + 1.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph         The phase's frozen graph.
 * @param[in]     source        Tail of the anchored edge.
 * @param[in]     target        Head of the anchored edge.
 * @param[in]     owner_id      Ownership id of the anchored edge in its phase.
 * @param[in]     phase_changes Ownership index of the phase.
 * @param[in]     max_length    Longest counted length (>= 2).
 * @param[in,out] scratch       The thread's scratch: marks (>= graph.vertex_count entries, all
 *                              zero, and zero again on a normal return), stack and counters.
 * @return The longest length counted (0 if none).
 * @throws std::bad_alloc if the stack or the counters cannot grow (the marks are then dirty: the
 *         caller restores the scratch with cycle_count_thread::reset()).
 */
template <typename vertex_t, typename edge_t>
std::size_t count_cycles_through_edge(const cycle_graph<vertex_t, edge_t>& graph,
                                      const vertex_t source, const vertex_t target,
                                      const std::size_t owner_id,
                                      const changed_edge_index& phase_changes,
                                      const std::size_t max_length,
                                      cycle_count_thread<vertex_t>& scratch) {
  // A self-loop lies on no simple cycle of length >= 2. The original's prepare_batch() drops
  // self-loop changes; under batch_semantics::as_sets with self_loop::keep the normalized lists
  // keep them (the graph stores the loop), so the change is skipped here without renumbering the
  // ids (a self-loop's id never decides the ownership of a cycle).
  if (max_length < 2 || source == target ||
      static_cast<std::size_t>(target) >= graph.vertex_count ||
      static_cast<std::size_t>(source) >= graph.vertex_count) {
    return 0;
  }
  char* const visited = scratch.visited.data();
  const edge_t* const offsets = graph.offsets;
  const vertex_t* const neighbors = graph.neighbors;
  std::vector<cycle_search_frame<vertex_t>>& stack = scratch.stack;
  if (stack.empty()) {
    cycle_count_grow_stack(stack);
  }
  cycle_search_frame<vertex_t>* frames = stack.data();
  std::size_t capacity = stack.size();
  std::uint64_t* counts = scratch.partial.data();
  std::size_t count_size = scratch.partial.size();
  std::size_t reached = 0;

  visited[static_cast<std::size_t>(source)] = 1;
  visited[static_cast<std::size_t>(target)] = 1;
  // The vertex being expanded and its cursor stay in locals (registers), as the recursion's do;
  // frames[d - 1] holds the suspended cursor of the path's d-th vertex while a deeper one is
  // expanded.
  vertex_t current = target;
  std::size_t next = static_cast<std::size_t>(offsets[target]);
  std::size_t end = static_cast<std::size_t>(offsets[target + 1]);
  std::size_t depth = 1;  // vertices on the path from the anchored target

  for (;;) {
    // Scan the row of the vertex being expanded up to the next vertex to extend the path with
    // (the order and the tests of search()'s loop).
    vertex_t v = 0;
    bool extend = false;
    while (next != end) {
      v = neighbors[next];
      ++next;
      if (phase_changes.forbidden_before(current, v, owner_id)) {
        continue;  // owned by a smaller-id changed edge
      }
      if (v == source) {
        const std::size_t length = depth + 1;
        if (length >= 2 && length <= max_length) {
          if (length >= count_size) {
            cycle_count_grow(scratch.partial, length, max_length);
            counts = scratch.partial.data();
            count_size = scratch.partial.size();
          }
          counts[length] += 1;
          reached = length > reached ? length : reached;
        }
        continue;
      }
      if (v == target || visited[static_cast<std::size_t>(v)] != 0) {
        continue;
      }
      if (depth + 1 < max_length) {
        extend = true;
        break;
      }
    }

    if (!extend) {  // the row is done: back to the parent
      if (--depth == 0) {
        break;  // the anchored target: its mark is cleared below
      }
      visited[static_cast<std::size_t>(current)] = 0;
      current = frames[depth - 1].vertex;
      next = frames[depth - 1].next;
      end = frames[depth - 1].end;
      continue;
    }

    if (depth > capacity) {
      cycle_count_grow_stack(stack);
      frames = stack.data();
      capacity = stack.size();
    }
    frames[depth - 1].vertex = current;
    frames[depth - 1].next = next;
    frames[depth - 1].end = end;
    visited[static_cast<std::size_t>(v)] = 1;
    current = v;
    next = static_cast<std::size_t>(offsets[v]);
    end = static_cast<std::size_t>(offsets[v + 1]);
    ++depth;
  }

  visited[static_cast<std::size_t>(source)] = 0;
  visited[static_cast<std::size_t>(target)] = 0;
  return reached;
}

}  // namespace dyng::detail
