// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/core/histogram.hpp (CycleHistogram)
// and include/cycle_enum/dynamic/update_sequential.hpp (update_static_histogram,
// apply_histogram_delta)
/**
 * @file problem.hpp
 * @brief The cycle_count problem: state, workspace and the hook entry points shared by the
 *        sequential and OpenMP backends.
 *
 * Template card (PLAN Section 4.5.2):
 *
 *     normalize -> translate -> prepare -> [before_apply -> (AG: count -)] -> commit ->
 *     identify_affected -> seed -> { AG: count + } -> finalize
 *
 * cycle_count is an aggregate-delta problem (family::aggregate_delta). Its hooks, one profiler
 * stage each:
 *   - normalize (cycle_count.normalize, on G_t): the net structural change of the batch
 *     (graph/structural_change.hpp, CycleEnumeration-GPU's prepare_batch);
 *   - before_apply = count(-) (cycle_count.count_minus, on G_t): the cycles through the deleted
 *     edges, each attributed to the deleted edge with the smallest id on it
 *     (ownership::min_member over the change index);
 *   - commit (cycle_count.commit): graph::apply under the graph's batch_semantics;
 *   - identify_affected (cycle_count.identify_affected): the inserted edges and their ownership
 *     index;
 *   - count(+) (cycle_count.count_plus, on G_{t+1}): the cycles through the inserted edges, each
 *     attributed to the inserted edge with the smallest id on it;
 *   - finalize (cycle_count.finalize): the signed delta applied to the histogram
 *     (apply_histogram_delta; internal_error if a bucket would go negative).
 * compute() is the static enactor: reset (cycle_count.reset) -> count (cycle_count.count) ->
 * finalize (cycle_count.finalize).
 *
 * The code stays organized by these hooks so that M3 can extract the framework (count_delta with
 * an ownership policy) without rewriting the engines.
 */
#pragma once

#include "framework/workspace.hpp"
#include "graph/structural_change.hpp"

#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/cycle_count.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail {

/**
 * @brief The state behind cycle_count::result.
 */
struct cycle_count_state {
  cycle_count::options opt;           ///< options given at compute()
  std::int64_t bound = 2;             ///< longest counted length (max_length, or max(n, 2))
  std::vector<std::uint64_t> counts;  ///< counts[len], size bound + 1
  std::uint64_t version = 0;          ///< graph version matched
  std::uint64_t graph_state = 0;      ///< state identifier of the graph matched (graph_impl)
  bool poisoned = false;              ///< a failed update left the histogram inconsistent
};

/**
 * @brief Internal access to cycle_count::result (for the implementation and the tests).
 */
struct cycle_count_access {
  /**
   * @brief The state of a result.
   * @param[in] r The result.
   * @return Its state (throws invalid_argument_error for a moved-from result).
   */
  static cycle_count_state& state(cycle_count::result& r);

  /**
   * @brief The state of a result (read-only).
   * @param[in] r The result.
   * @return Its state (throws invalid_argument_error for a moved-from result).
   */
  static const cycle_count_state& state(const cycle_count::result& r);

  /**
   * @brief Wrap a state into a result.
   * @param[in] state The state.
   * @return The result.
   */
  static cycle_count::result make(std::unique_ptr<cycle_count_state> state);
};

/**
 * @brief The out-edges as the engines read them (CycleEnumeration-GPU's DirectedGraph rows):
 *        sorted rows without parallel edges.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct cycle_graph {
  std::size_t vertex_count = 0;         ///< n
  const edge_t* offsets = nullptr;      ///< n + 1 row offsets
  const vertex_t* neighbors = nullptr;  ///< sorted out-neighbours per row
};

/**
 * @brief Maps a phase's changed edges to their ownership ids (CycleEnumeration-GPU's
 *        ChangedEdgeIndex): the id of a changed edge is its index in the normalized (sorted)
 *        change list; a cycle is counted only by the smallest-id changed edge it contains.
 *
 * The original keeps a std::unordered_map, which allocates one node per changed edge on every
 * update; dynG keeps a flat open-addressing table (linear probing, load factor <= 1/2) whose
 * arrays are reused, so rebuilding it for a batch no larger than an earlier one allocates nothing
 * (invariant I9). The lookups answer exactly as the map's.
 */
class changed_edge_index {
 public:
  /**
   * @brief Rebuild the index from a normalized change list (index = id); the table's capacity is
   *        reused.
   * @tparam vertex_t Vertex id type (32-bit).
   * @param[in] changes The change list (distinct edges).
   */
  template <typename vertex_t>
  void assign(const std::vector<edge_change<vertex_t>>& changes) {
    static_assert(sizeof(vertex_t) == 4, "changed_edge_index keys two 32-bit vertex ids");
    std::size_t slots = min_slots;
    int bits = min_bits;
    while (slots < 2 * changes.size()) {
      slots *= 2;
      ++bits;
    }
    if (keys_.size() < slots) {
      keys_.assign(slots, empty_key);
      ids_.assign(slots, 0);
    } else {
      std::fill(keys_.begin(), keys_.begin() + static_cast<std::ptrdiff_t>(slots), empty_key);
    }
    mask_ = slots - 1;
    shift_ = 64 - bits;
    size_ = changes.size();
    for (std::size_t index = 0; index < changes.size(); ++index) {
      const std::uint64_t k = key(changes[index].source, changes[index].target);
      std::size_t slot = home(k);
      while (keys_[slot] != empty_key) {
        slot = (slot + 1) & mask_;
      }
      keys_[slot] = k;
      ids_[slot] = index;
    }
  }

  /**
   * @brief Whether (source, target) is a changed edge whose id is strictly smaller than
   *        `owner_id` (a traversal anchored at the edge `owner_id` must not cross it: every cycle
   *        through it belongs to the smaller id).
   * @tparam vertex_t Vertex id type (32-bit).
   * @param[in] source   Tail of the edge.
   * @param[in] target   Head of the edge.
   * @param[in] owner_id Ownership id of the anchored edge.
   * @return true if the edge is owned by a smaller id.
   */
  template <typename vertex_t>
  [[nodiscard]] bool forbidden_before(vertex_t source, vertex_t target,
                                      std::size_t owner_id) const noexcept {
    if (size_ == 0) {
      return false;
    }
    const std::uint64_t k = key(source, target);
    for (std::size_t slot = home(k);; slot = (slot + 1) & mask_) {
      const std::uint64_t found = keys_[slot];
      if (found == k) {
#if defined(DYNG_MUTATION_WEAK_OWNERSHIP)
        // Recorded mutation (PLAN Section 8.4, "weakened ownership rule"): the changed edge with
        // the id just below the anchor no longer takes precedence, so cycles through two
        // consecutive changed edges are counted twice. The mutation tests require the suite to
        // fail.
        return ids_[slot] + 1 < owner_id;
#else
        return ids_[slot] < owner_id;
#endif
      }
      if (found == empty_key) {
        return false;
      }
    }
  }

  /**
   * @brief Number of indexed changed edges.
   * @return The number of edges of the last assign().
   */
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  /**
   * @brief The memory the table holds.
   * @return Bytes of its arrays' capacity.
   */
  [[nodiscard]] std::size_t bytes() const noexcept {
    return keys_.capacity() * sizeof(std::uint64_t) + ids_.capacity() * sizeof(std::size_t);
  }

 private:
  static constexpr int min_bits = 4;
  static constexpr std::size_t min_slots = std::size_t{1} << min_bits;
  /// No edge has this key: vertex ids are non-negative 32-bit values below 2^31.
  static constexpr std::uint64_t empty_key = ~std::uint64_t{0};

  template <typename vertex_t>
  static std::uint64_t key(vertex_t source, vertex_t target) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(target));
  }

  /// Fibonacci hashing: the top bits of key * 2^64 / phi.
  [[nodiscard]] std::size_t home(std::uint64_t k) const noexcept {
    return static_cast<std::size_t>((k * 0x9E3779B97F4A7C15ULL) >> shift_) & mask_;
  }

  std::vector<std::uint64_t> keys_;  ///< empty_key or the key of the edge in the slot
  std::vector<std::size_t> ids_;     ///< the ownership id of the edge in the slot
  std::size_t mask_ = 0;             ///< slots in use - 1 (a power of two minus one)
  int shift_ = 64;                   ///< 64 - log2(slots in use)
  std::size_t size_ = 0;             ///< indexed edges
};

/**
 * @brief Scratch space of update(): leased from the workspace pool of the resources handle for
 *        one update and shared by every result updated through that handle (ADR 0015).
 *
 * `visited` replaces the original's `static thread_local` visited buffer of
 * count_cycles_through_edge() (no globals): one array per thread, all zero between searches (the
 * search clears every mark it sets). The other arrays are the per-thread histograms of a phase
 * and the phase totals.
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct cycle_count_workspace final : pooled_workspace {
  structural_change<vertex_t> change;      ///< the normalized batch (Step 0)
  changed_edge_index index;                ///< ownership ids of the current phase
  std::vector<std::vector<char>> visited;  ///< per thread: path marks, n entries, all zero
  std::size_t marks_needed = 0;            ///< entries each thread's marks must have
  /// Elements per thread in `counts` and `partial`: max_length + 1 rounded up to a cache line,
  /// plus one line, so no two threads write to one cache line.
  std::size_t stride = 0;
  std::vector<std::uint64_t> counts;   ///< per thread (at t * stride): cycles through one edge
  std::vector<std::uint64_t> partial;  ///< per thread (at t * stride): cycles of the phase
  std::vector<std::uint64_t> removed;  ///< cycles through deleted edges, by length
  std::vector<std::uint64_t> added;    ///< cycles through inserted edges, by length

  /**
   * @brief Size the per-thread arrays (no-op if large enough; `visited` keeps its zeros).
   * @param[in] threads    Threads of the phase.
   * @param[in] vertices   Vertices of the largest graph searched.
   * @param[in] max_length Longest counted length.
   */
  void reserve(int threads, std::size_t vertices, std::size_t max_length);

  /**
   * @brief The path marks of one thread, sized on first use by the thread that calls this (so the
   *        pages are first touched in parallel, as the original's thread_local buffers are).
   * @param[in] thread The thread.
   * @return At least marks_needed entries, all zero.
   */
  [[nodiscard]] std::vector<char>& thread_visited(std::size_t thread) {
    std::vector<char>& marks = visited[thread];
    if (marks.size() < marks_needed) {
      marks.resize(marks_needed, 0);
    }
    return marks;
  }

  /**
   * @brief The per-edge counts of one thread.
   * @param[in] thread The thread.
   * @return max_length + 1 entries.
   */
  [[nodiscard]] std::uint64_t* thread_counts(std::size_t thread) noexcept {
    return counts.data() + thread * stride;
  }

  /**
   * @brief The phase sums of one thread.
   * @param[in] thread The thread.
   * @return max_length + 1 entries.
   */
  [[nodiscard]] std::uint64_t* thread_partial(std::size_t thread) noexcept {
    return partial.data() + thread * stride;
  }

  /**
   * @brief The memory the workspace holds.
   * @return Bytes of every array's capacity.
   */
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/**
 * @brief Sequential backend, compute(): CycleEnumeration-GPU's sequential Johnson
 *        (count_simple_cycles_johnson).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]  graph      The graph.
 * @param[in]  max_length Longest counted length (>= 2), or -1 for no bound.
 * @param[out] counts     counts[len] (size >= longest possible length + 1, zero on entry).
 */
template <typename vertex_t, typename edge_t>
void cycle_count_sequential_compute(const cycle_graph<vertex_t, edge_t>& graph,
                                    std::int64_t max_length, std::vector<std::uint64_t>& counts);

/**
 * @brief OpenMP backend, compute(): CycleEnumeration-GPU's OpenMP counter
 *        (openmp::count_simple_cycles_johnson): roots in parallel, one histogram per thread.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]  graph      The graph.
 * @param[in]  max_length Longest counted length (>= 2), or -1 for no bound.
 * @param[in]  threads    OpenMP threads (1: the single-thread loop of the original).
 * @param[out] counts     counts[len] (size >= longest possible length + 1, zero on entry).
 * @throws capacity_error if a merged count exceeds 2^64 - 1.
 */
template <typename vertex_t, typename edge_t>
void cycle_count_openmp_compute(const cycle_graph<vertex_t, edge_t>& graph, std::int64_t max_length,
                                int threads, std::vector<std::uint64_t>& counts);

/**
 * @brief Sequential backend, one phase of update() (accumulate_phase of
 *        update_static_histogram): the cycles through every change edge that it owns.
 *
 * Precondition: `ws.index` holds the ownership ids of `changes`, and `ws` is reserved for one
 * thread, the vertices of `graph` and `max_length`.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph      The phase's graph (G_t for the deletions, G_{t+1} for the insertions).
 * @param[in]     changes    The phase's normalized change list.
 * @param[in]     max_length Longest counted length (>= 2).
 * @param[in,out] ws         The workspace.
 * @param[out]    phase      phase[len] += the owned cycles of length len (size max_length + 1).
 */
template <typename vertex_t, typename edge_t>
void cycle_count_sequential_phase(const cycle_graph<vertex_t, edge_t>& graph,
                                  const std::vector<edge_change<vertex_t>>& changes,
                                  std::size_t max_length, cycle_count_workspace<vertex_t>& ws,
                                  std::vector<std::uint64_t>& phase);

/**
 * @brief OpenMP backend, one phase of update() (accumulate_phase_parallel of
 *        update_static_histogram_openmp): change edges in parallel, one histogram per thread.
 *
 * Precondition: as cycle_count_sequential_phase(), with `ws` reserved for `threads` threads.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph      The phase's graph.
 * @param[in]     changes    The phase's normalized change list.
 * @param[in]     max_length Longest counted length (>= 2).
 * @param[in]     threads    OpenMP threads (> 1).
 * @param[in,out] ws         The workspace.
 * @param[out]    phase      phase[len] += the owned cycles of length len (size max_length + 1).
 * @throws capacity_error if a merged count exceeds 2^64 - 1.
 */
template <typename vertex_t, typename edge_t>
void cycle_count_openmp_phase(const cycle_graph<vertex_t, edge_t>& graph,
                              const std::vector<edge_change<vertex_t>>& changes,
                              std::size_t max_length, int threads,
                              cycle_count_workspace<vertex_t>& ws,
                              std::vector<std::uint64_t>& phase);

/**
 * @brief Record one cycle of `length` in a static count (CycleHistogram::increment).
 *
 * Built with DYNG_MUTATION_DOUBLE_COUNT_5 it is the recorded mutation "double counting 5-cycles"
 * (PLAN Section 8.4): 5-cycles count twice, and the mutation tests require the suite to fail.
 * @param[in,out] counts The histogram.
 * @param[in]     length The cycle length (< counts.size()).
 */
inline void cycle_count_record(std::vector<std::uint64_t>& counts, std::size_t length) {
#if defined(DYNG_MUTATION_DOUBLE_COUNT_5)
  counts[length] += length == 5 ? 2 : 1;
#else
  counts[length] += 1;
#endif
}

/**
 * @brief a += b, throwing capacity_error on overflow (CycleHistogram's ensure_addition_safe).
 * @param[in,out] a The sum.
 * @param[in]     b The addend.
 * @throws capacity_error if the sum exceeds 2^64 - 1.
 */
void cycle_count_checked_add(std::uint64_t& a, std::uint64_t b);

/**
 * @brief Run one hook inside its profiler stage `cycle_count.<hook>`.
 * @tparam fn_t Callable.
 * @param[in] res  Resources (profiler).
 * @param[in] name Stage name.
 * @param[in] fn   The hook.
 */
template <typename fn_t>
void cycle_count_hook(const resources& res, const char* name, fn_t&& fn) {
  scoped_stage stage(res, name);
  fn();
}

}  // namespace dyng::detail
