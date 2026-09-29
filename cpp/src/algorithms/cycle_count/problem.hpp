// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/core/histogram.hpp (CycleHistogram)
// and include/cycle_enum/dynamic/update_sequential.hpp (update_static_histogram,
// apply_histogram_delta)
/**
 * @file problem.hpp
 * @brief The cycle_count problem on the framework (cpp/src/framework): its hooks, its state and
 *        workspaces, and the engines of the three backends.
 *
 * Template card (PLAN Section 4.5.2):
 *
 *     normalize -> translate -> prepare -> [before_apply -> (AG: count -)] -> commit ->
 *     identify_affected -> seed -> { AG: count + } -> finalize
 *
 * cycle_count is an aggregate-delta problem (cycle_count_problem below, family::aggregate_delta,
 * ownership_type = ownership::min_member; its members are defined in cycle_count.cpp, which runs
 * it through framework::update_enactor and framework::static_enactor). Its hooks, one profiler
 * stage each:
 *   - begin_update (no stage): the backend, poisoned and stale-result checks, the graph's
 *     requirements, the placement and the CUDA engine; the lease of the host workspace;
 *   - normalize (cycle_count.normalize, on G_t): the net structural change of the batch
 *     (graph/structural_change.hpp, CycleEnumeration-GPU's prepare_batch), or under
 *     batch_semantics::as_sets the framework's lists of Step 0 (run_update(), ADR 0020) as the
 *     engines' change lists; on cuda the check that the bound after the batch fits the device
 *     counters;
 *   - count(-) on the old view (cycle_count.count_minus, on G_t): the cycles through the deleted
 *     edges, each attributed to the deleted edge with the smallest id on it
 *     (ownership::min_member over the change index, changed_edge_index);
 *   - commit (cycle_count.commit, run_update()): graph::apply under the graph's batch_semantics;
 *   - resume (no stage): the Debug check of the normalized batch, the bound after the batch;
 *   - identify_affected (cycle_count.identify_affected): the inserted edges and their ownership
 *     index;
 *   - count(+) on the new view (cycle_count.count_plus, on G_{t+1}): the cycles through the
 *     inserted edges, each attributed to the inserted edge with the smallest id on it;
 *   - finalize (cycle_count.finalize): the signed delta applied to the histogram
 *     (apply_histogram_delta; internal_error if a bucket would go negative), the stats;
 *   - end_update (no stage): record the graph state the result matches, return the workspaces.
 * compute() is the static enactor: reset (cycle_count.reset) -> count (cycle_count.count) ->
 * finalize (cycle_count.finalize).
 *
 * The host backends run these hooks (Tier A) with the engines of sequential.cpp / openmp.cpp
 * (static_sequential.cpp / static_openmp.cpp for compute()).
 */
#pragma once

#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"
#include "graph/structural_change.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dyng::detail {

/**
 * @brief The state behind cycle_count::result.
 */
struct cycle_count_state {
  cycle_count::options opt;           ///< options given at compute()
  std::int64_t bound = 2;             ///< min(max_length, max(n, 2)); max(n, 2) without a bound
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
 * @brief One level of an explicit depth-first search: a vertex on the path and its next
 *        unexplored out-edge (the searches are iterative, so their depth is bounded by memory, not
 *        by the thread's stack).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct cycle_search_frame {
  vertex_t vertex{};     ///< the vertex on the path
  std::size_t next = 0;  ///< offset of its next out-edge to explore
  std::size_t end = 0;   ///< end of its row
};

/**
 * @brief Grow a per-thread counts array so that `length` is a valid index (at least doubling, at
 *        most `max_length` + 1 entries; new entries are zero).
 * @param[in,out] counts     The array.
 * @param[in]     length     The index needed (<= max_length).
 * @param[in]     max_length Longest counted length.
 */
inline void cycle_count_grow(std::vector<std::uint64_t>& counts, std::size_t length,
                             std::size_t max_length) {
  const std::size_t wanted = std::max(length + 1, 2 * counts.size());
  counts.resize(std::min(wanted, max_length + 1), 0);
}

/**
 * @brief Grow a search stack (at least doubling; new frames are default-constructed).
 * @tparam frame_t Frame type.
 * @param[in,out] stack The stack.
 */
template <typename frame_t>
void cycle_count_grow_stack(std::vector<frame_t>& stack) {
  stack.resize(std::max<std::size_t>(16, 2 * stack.size()));
}

/**
 * @brief The scratch of one thread of update() (alignas: no two threads share a cache line).
 *
 * Between phases every mark and every counter is zero: the search clears every mark it sets, and
 * the reduction clears the counters it reads.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct alignas(64) cycle_count_thread {
  std::vector<char> visited;                        ///< path marks, >= marks_needed entries
  std::vector<cycle_search_frame<vertex_t>> stack;  ///< the explicit search stack
  std::vector<std::uint64_t> partial;  ///< cycles of the phase by length (grown on demand)
  std::size_t reached = 0;             ///< longest length counted in `partial` this phase

  /**
   * @brief Size the marks and the counters (called by the thread itself, so the pages are first
   *        touched in parallel, as the original's thread_local buffers are).
   * @param[in] marks   Entries the marks need (the vertex count of the phase's graph).
   * @param[in] lengths Initial entries of the counters.
   */
  void prepare(std::size_t marks, std::size_t lengths) {
    if (visited.size() < marks) {
      visited.resize(marks, 0);
    }
    if (partial.size() < lengths) {
      partial.reserve(lengths + 8);  // a spare cache line after the counters
      partial.resize(lengths, 0);
    }
    if (stack.empty()) {
      cycle_count_grow_stack(stack);
    }
    reached = 0;
  }

  /**
   * @brief Restore the invariant (all marks and counters zero) after a failed phase.
   */
  void reset() noexcept {
    std::fill(visited.begin(), visited.end(), 0);
    std::fill(partial.begin(), partial.end(), 0);
    reached = 0;
  }
};

/**
 * @brief Scratch space of update(): leased from the workspace pool of the resources handle for
 *        one update and shared by every result updated through that handle (ADR 0015).
 *
 * `threads[t].visited` replaces the original's `static thread_local` visited buffer of
 * count_cycles_through_edge() (no globals). The per-thread counters grow with the longest cycle
 * found, not with the length bound (so an unbounded update costs what its searches cost, not
 * O(n) per change edge).
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct cycle_count_workspace final : pooled_workspace {
  structural_change<vertex_t> change;                 ///< the normalized batch (Step 0)
  changed_edge_index index;                           ///< ownership ids of the current phase
  std::vector<cycle_count_thread<vertex_t>> threads;  ///< per-thread scratch
  std::size_t marks_needed = 0;                       ///< entries each thread's marks must have
  std::vector<std::uint64_t> removed;                 ///< cycles through deleted edges, by length
  std::vector<std::uint64_t> added;                   ///< cycles through inserted edges, by length

  /**
   * @brief Make room for `threads` threads searching a graph of `vertices` vertices (the arrays
   *        themselves are sized by their thread inside the phase).
   * @param[in] thread_count Threads of the phase.
   * @param[in] vertices     Vertices of the largest graph searched.
   */
  void reserve(int thread_count, std::size_t vertices);

  /**
   * @brief Restore the invariant of every thread's scratch after a failed phase.
   */
  void reset_threads() noexcept {
    for (cycle_count_thread<vertex_t>& t : threads) {
      t.reset();
    }
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
 * @param[out] counts     counts[len] (min(max_length, max(n, 2)) + 1 entries, zero on entry).
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
 * @param[out] counts     counts[len] (min(max_length, max(n, 2)) + 1 entries, zero on entry).
 * @throws capacity_error if a merged count exceeds 2^64 - 1.
 * @throws std::bad_alloc if a thread's scratch cannot be allocated.
 */
template <typename vertex_t, typename edge_t>
void cycle_count_openmp_compute(const cycle_graph<vertex_t, edge_t>& graph, std::int64_t max_length,
                                int threads, std::vector<std::uint64_t>& counts);

/**
 * @brief Sequential backend, one phase of update() (accumulate_phase of
 *        update_static_histogram): the cycles through every change edge that it owns.
 *
 * Precondition: `ws.index` holds the ownership ids of `changes`, and `ws` is reserved for one
 * thread and the vertices of `graph`. On an exception the workspace's invariant is restored.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph      The phase's graph (G_t for the deletions, G_{t+1} for the insertions).
 * @param[in]     changes    The phase's normalized change list.
 * @param[in]     max_length Longest counted length (>= 2; at most max(n, 2) is useful).
 * @param[in,out] ws         The workspace.
 * @param[in,out] phase      phase[len] += the owned cycles of length len (grown to the longest
 *                           length found + 1 if shorter).
 * @throws capacity_error if a count exceeds 2^64 - 1.
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
 * @param[in]     max_length Longest counted length (>= 2; at most max(n, 2) is useful).
 * @param[in]     threads    OpenMP threads (> 1).
 * @param[in,out] ws         The workspace.
 * @param[in,out] phase      phase[len] += the owned cycles of length len (grown to the longest
 *                           length found + 1 if shorter).
 * @throws capacity_error if a merged count exceeds 2^64 - 1.
 * @throws std::bad_alloc if a thread's scratch cannot be allocated.
 */
template <typename vertex_t, typename edge_t>
void cycle_count_openmp_phase(const cycle_graph<vertex_t, edge_t>& graph,
                              const std::vector<edge_change<vertex_t>>& changes,
                              std::size_t max_length, int threads,
                              cycle_count_workspace<vertex_t>& ws,
                              std::vector<std::uint64_t>& phase);

/**
 * @brief The reduction of a phase: add the counters of the first `thread_count` threads into
 *        `phase` (grown to the longest length reached + 1 if shorter) and clear them.
 *
 * Only the lengths each thread reached are read and cleared, so the cost does not depend on the
 * length bound. On an exception every thread's scratch is reset.
 * @tparam vertex_t Vertex id type.
 * @param[in,out] ws           The workspace.
 * @param[in]     thread_count Threads of the phase.
 * @param[in,out] phase        The phase's histogram.
 * @throws capacity_error if a sum exceeds 2^64 - 1.
 */
template <typename vertex_t>
void cycle_count_drain(cycle_count_workspace<vertex_t>& ws, std::size_t thread_count,
                       std::vector<std::uint64_t>& phase);

/**
 * @brief Record one cycle of `length` in a static count (CycleHistogram::increment).
 *
 * Built with DYNG_MUTATION_DOUBLE_COUNT_5 it is the recorded mutation "double counting 5-cycles"
 * (PLAN Section 8.4): 5-cycles count twice, and the mutation tests require the suite to fail.
 * @param[in,out] counts The histogram.
 * @param[in]     length The cycle length (a valid index of `counts`).
 */
inline void cycle_count_record(std::uint64_t* counts, std::size_t length) noexcept {
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

// ------------------------------------------------------------------------------------------------
// The CUDA backend (static_cuda.cu, cuda.cu; CycleEnumeration-GPU's src/cuda and
// src/dynamic/update_cuda*)
// ------------------------------------------------------------------------------------------------

/**
 * @brief The out-edges of a device graph as the CUDA engines read them (device pointers).
 * @tparam edge_t Edge offset type.
 */
template <typename edge_t>
struct cycle_device_graph {
  std::int64_t vertex_count = 0;            ///< n
  std::int64_t edge_count = 0;              ///< m
  const edge_t* offsets = nullptr;          ///< n + 1 row offsets (device)
  const std::int32_t* neighbors = nullptr;  ///< sorted out-neighbours (device)
};

/**
 * @brief Scratch of the CUDA backend, leased from the workspace pool of the resources handle for
 *        one compute() or update() (ADR 0015): the arrays the original allocates per call
 *        (DeviceBuffer), kept across calls.
 * @tparam edge_t Edge offset type.
 */
template <typename edge_t>
struct cycle_count_cuda_workspace final : pooled_workspace {
  scratch_buffer<std::make_unsigned_t<edge_t>> forward_begin;   ///< static: split of each row
  scratch_buffer<std::make_unsigned_t<edge_t>> forward_count;   ///< static: forward degrees
  scratch_buffer<std::make_unsigned_t<edge_t>> forward_offset;  ///< static: their scan
  scratch_buffer<std::uint32_t> item_source;                    ///< static: edge items (tails)
  scratch_buffer<std::uint32_t> item_target;                    ///< static: edge items (heads)
  scratch_buffer<unsigned long long> hop_count;                 ///< static: two-hop counts
  scratch_buffer<unsigned long long> hop_offset;                ///< static: their scan
  scratch_buffer<std::int32_t> owner;                ///< update: ownership id per edge position
  scratch_buffer<unsigned long long> item_counts;    ///< update: items per change
  scratch_buffer<unsigned long long> item_offsets;   ///< update: their scan
  scratch_buffer<std::uint32_t> changes;             ///< update: own change lists (pairs)
  scratch_buffer<unsigned long long> histograms;     ///< 2 x 65 counts (deleted, inserted)
  scratch_buffer<unsigned long long> work_counters;  ///< the queues' global counters
  scratch_buffer<unsigned char> scan_temp;           ///< CUB scratch
  buffer<unsigned long long> host_scalars;           ///< pinned: 2 x 65 counts + read-back scalars
  buffer<std::uint32_t> host_changes;                ///< pinned: staging of `changes`
  int sm_count = 0;  ///< multiprocessors of the device (0: not queried)

  /**
   * @brief The memory held.
   * @return Bytes of every buffer.
   */
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return forward_begin.bytes() + forward_count.bytes() + forward_offset.bytes() +
           item_source.bytes() + item_target.bytes() + hop_count.bytes() + hop_offset.bytes() +
           owner.bytes() + item_counts.bytes() + item_offsets.bytes() + changes.bytes() +
           histograms.bytes() + work_counters.bytes() + scan_temp.bytes() +
           host_scalars.size() * sizeof(unsigned long long) +
           host_changes.size() * sizeof(std::uint32_t);
  }
};

/**
 * @brief CUDA backend, compute(): CycleEnumeration-GPU's count_simple_cycles_johnson_device()
 *        (options::scheduler naive) or count_simple_cycles_johnson_queue_device() (work queue).
 *
 * Profiler stage cycle_count.count: building the work items and counting (the original's
 * kernel_ms region); the histogram is copied back in cycle_count.finalize.
 * @tparam edge_t Edge offset type.
 * @param[in]     res    Resources of the CUDA backend.
 * @param[in]     graph  The resident out-edges.
 * @param[in]     length The effective length bound (<= 64).
 * @param[in]     opt    Options (scheduler, work items).
 * @param[in,out] ws     The workspace.
 * @param[out]    counts counts[len] for len <= length (zero on entry, >= length + 1 entries).
 */
template <typename edge_t>
void cycle_count_cuda_compute(const resources& res, const cycle_device_graph<edge_t>& graph,
                              std::int64_t length, const cycle_count::options& opt,
                              cycle_count_cuda_workspace<edge_t>& ws,
                              std::vector<std::uint64_t>& counts);

/**
 * @brief CUDA backend, one phase of update() (run_phase of count_update_cycles_device): the
 *        cycles each change owns in `graph`, added to device histogram `slot` (0: the deletions,
 *        1: the insertions). Asynchronous except for the read-back of the item count.
 *
 * `owner` holds, per edge position of `graph`, the id of the change stored there or
 * no_change_id; nullptr: it is built here (mark_owners_kernel on the workspace's owner array).
 * @tparam edge_t Edge offset type.
 * @param[in]     res          Resources of the CUDA backend.
 * @param[in]     graph        The phase's resident graph (G_t or G_{t+1}).
 * @param[in]     owner        The ownership array, or nullptr.
 * @param[in]     changes      The phase's change list on the device ((source, target) pairs).
 * @param[in]     change_count Number of changes.
 * @param[in]     length       The effective length bound (<= 64).
 * @param[in]     slot         0 or 1.
 * @param[in,out] ws           The workspace (histograms cleared by cycle_count_cuda_begin_update).
 */
template <typename edge_t>
void cycle_count_cuda_phase(const resources& res, const cycle_device_graph<edge_t>& graph,
                            const std::int32_t* owner, const std::uint32_t* changes,
                            std::uint32_t change_count, std::int64_t length, int slot,
                            cycle_count_cuda_workspace<edge_t>& ws);

/**
 * @brief Clear the two device histograms of an update (before its first phase).
 * @tparam edge_t Edge offset type.
 * @param[in]     res Resources of the CUDA backend.
 * @param[in,out] ws  The workspace.
 */
template <typename edge_t>
void cycle_count_cuda_begin_update(const resources& res, cycle_count_cuda_workspace<edge_t>& ws);

/**
 * @brief Upload a change list of the workspace's own (the batch was not normalized by the
 *        framework: batch semantics other than set()); returns the device pairs.
 * @tparam edge_t   Edge offset type.
 * @tparam vertex_t Vertex id type.
 * @param[in]     res       Resources of the CUDA backend.
 * @param[in]     deletions The deletions.
 * @param[in]     insertions The insertions (uploaded after the deletions).
 * @param[in,out] ws        The workspace.
 * @return Device pointer to the deletion pairs; the insertion pairs follow them.
 */
template <typename edge_t, typename vertex_t>
const std::uint32_t* cycle_count_cuda_upload_changes(
    const resources& res, const std::vector<edge_change<vertex_t>>& deletions,
    const std::vector<edge_change<vertex_t>>& insertions, cycle_count_cuda_workspace<edge_t>& ws);

/**
 * @brief Copy the two device histograms of an update back (synchronizes the stream).
 * @tparam edge_t Edge offset type.
 * @param[in]     res     Resources of the CUDA backend.
 * @param[in]     length  The effective length bound.
 * @param[in,out] ws      The workspace.
 * @param[out]    removed removed[len] for len <= length (resized to length + 1).
 * @param[out]    added   added[len] for len <= length (resized to length + 1).
 */
template <typename edge_t>
void cycle_count_cuda_end_update(const resources& res, std::int64_t length,
                                 cycle_count_cuda_workspace<edge_t>& ws,
                                 std::vector<std::uint64_t>& removed,
                                 std::vector<std::uint64_t>& added);

namespace framework {
template <typename vertex_t, typename weight_t>
struct requested_batch;
template <typename vertex_t>
struct applied_batch;
}  // namespace framework

/**
 * @brief The cycle_count problem (family::aggregate_delta) that framework::update_enactor and
 *        framework::static_enactor run: the hooks of the file comment. Its members are defined in
 *        cycle_count.cpp, the only translation unit that runs it.
 *
 * One problem serves one call: an update of one result (constructed from the result; the
 * participant adapter of framework/composition.hpp owns it) or a compute() (constructed from the
 * state being built, then bind_static()). From begin_update() to end_update() (or the problem's
 * destruction) it holds the lease of the pooled host workspace, whose change lists and ownership
 * index are the problem's frontier (internal_frontier).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class cycle_count_problem final
    : public framework::problem_base<cycle_count_problem<vertex_t, edge_t, weight_t>,
                                     framework::family::aggregate_delta> {
 public:
  static constexpr std::string_view name = "cycle_count";    ///< stages "cycle_count.<hook>"
  using container_type = graph<vertex_t, edge_t, weight_t>;  ///< the container
  using stats_type = cycle_count::stats;                     ///< the stats of update()
  using ownership_type = framework::ownership::min_member;   ///< the counting rule (I2)
  using old_graph = framework::old_view<container_type>;     ///< G_t
  using new_graph = framework::new_view<container_type>;     ///< G_{t+1}
  using requested = framework::requested_batch<vertex_t, weight_t>;  ///< the batch before commit
  using applied = framework::applied_batch<vertex_t>;                ///< what the commit did
  using frontier = framework::internal_frontier;                     ///< frontiers are internal
  using workspace_type = cycle_count_workspace<vertex_t>;            ///< the host scratch

  /**
   * @brief The problem of one update of `r`.
   * @param[in,out] r The result (must outlive the problem).
   */
  explicit cycle_count_problem(cycle_count::result& r) noexcept : result_(&r) {}

  /**
   * @brief The problem of one compute() that fills `st` (bind_static() follows).
   * @param[in,out] st The state of the result being built (must outlive the problem).
   */
  explicit cycle_count_problem(cycle_count_state& st) noexcept : state_(&st), computing_(true) {}

  // ---- lifecycle (no stage) ------------------------------------------------------------------

  /**
   * @brief The result this problem updates.
   * @return Its address (nullptr for compute()).
   */
  [[nodiscard]] const void* target() const noexcept {
    return result_;
  }

  /**
   * @brief The commit need not build the in-edges or the device copy: the engines read the
   *        out-edges only (on cuda graph_access::device_out, resident across batches).
   * @return false.
   */
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return false;
  }

  /**
   * @brief Validate the call before Step 0 (backend, poisoned and stale result, the graph's
   *        requirements, the placement, the CUDA engine) and lease the host workspace.
   * @throws not_supported_error, stale_result_error, invalid_argument_error.
   */
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);

  /**
   * @brief Re-bind to G_{t+1}: the Debug check of the normalized batch, the bound after the batch.
   */
  void resume(framework::context& ctx, new_graph g, const applied& applied);

  /**
   * @brief Record the graph state the result matches; return the workspaces.
   */
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);

  /**
   * @brief Mark the result unusable (its algorithm phase failed).
   */
  void poison() noexcept;

  /**
   * @brief compute(): bind the graph and the engine of the call's backend.
   * @param[in,out] ctx The run's context.
   * @param[in]     g   The graph.
   */
  void bind_static(framework::context& ctx, new_graph g);

  // ---- Step 0 and Step 1a, on G_t --------------------------------------------------------------

  /**
   * @brief cycle_count.normalize: the net structural change of the batch (or the framework's
   *        lists under as_sets).
   * @throws invalid_argument_error for a malformed batch, or on cuda a bound after the batch
   *         above 64.
   */
  void normalize(framework::context& ctx, old_graph g, const requested& batch);

  /**
   * @brief cycle_count.count_minus: the owned cycles through the deleted edges on G_t.
   * @throws capacity_error if a count exceeds 2^64 - 1.
   */
  void count(framework::context& ctx, old_graph g, frontier& f, framework::sign s,
             ownership_type rule);

  // ---- Tier A, on G_{t+1} ----------------------------------------------------------------------

  /// cycle_count.identify_affected: the inserted edges and their ownership index.
  void identify_affected(framework::context& ctx, new_graph g, const applied& applied, frontier& f);

  /**
   * @brief update(): cycle_count.count_plus, the owned cycles through the inserted edges on
   *        G_{t+1}; compute(): cycle_count.count, the static count of the whole graph.
   * @throws capacity_error if a count exceeds 2^64 - 1.
   */
  void count(framework::context& ctx, new_graph g, frontier& f, framework::sign s,
             ownership_type rule);

  /**
   * @brief update(): cycle_count.finalize, the signed delta applied to the histogram and the
   *        stats; compute(): cycle_count.finalize, the graph state the histogram matches.
   * @throws internal_error if a bucket would become negative (nothing is written then).
   * @throws capacity_error if a count exceeds 2^64 - 1.
   */
  void finalize(framework::context& ctx, stats_type& stats);

  // ---- compute(), static enactor ---------------------------------------------------------------

  /// cycle_count.reset: the zero histogram of the bound.
  void reset(framework::context& ctx);

 private:
  /// One host phase: the OpenMP phase with more than one thread, the sequential phase otherwise
  /// (update_static_histogram_openmp()).
  void run_host_phase(const container_type& g, const std::vector<edge_change<vertex_t>>& changes,
                      std::size_t max_length, std::vector<std::uint64_t>& phase);
  /// The counters of the update in the stats (all but the histogram delta).
  void fill_counts(stats_type& stats) const;

  cycle_count::result* result_ = nullptr;         ///< update(): the result
  cycle_count_state* state_ = nullptr;            ///< its state (begin_update) or compute()'s
  bool computing_ = false;                        ///< the problem serves a compute()
  const container_type* static_graph_ = nullptr;  ///< compute(): the graph
  int threads_ = 1;                               ///< host threads of the phases
  bool cuda_ = false;                             ///< the call runs on the cuda backend
  std::int64_t bound_before_ = 2;                 ///< the histogram bound of G_t
  std::int64_t bound_after_ = 2;                  ///< the histogram bound of G_{t+1}
  std::int64_t device_length_ = 2;                ///< cuda: the bound of the device phases
  const normalized_batch<vertex_t>* normalized_ = nullptr;   ///< the framework's Step 0, if any
  std::optional<workspace_pool::lease<workspace_type>> ws_;  ///< the host scratch
};

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
