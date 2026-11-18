// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (halfSweep, halfOffset,
// appendToQueue, appendNextFrontier, initFrontierKernel's queue) and
// headers/iterativePropagation.h (the degree bins, the queue sets)
/**
 * @file binned_frontier.hpp
 * @brief binned_frontier<id_t, num_bins> (PLAN 4.5.3): a frontier held as one queue per degree bin
 *        and half-sweep, in two sets that alternate between iterations, deduplicated by run-wide
 *        iteration stamps; its host half (the executor-templated calls and the sequential push).
 *        The CUDA half (the warp-aggregated push, expand on cuda_exec) is binned_frontier.cuh.
 *
 * The layout is the original's: a set holds, per bin b, one array of capacity capacity[b] whose
 * half-sweep 0 is filled from the front and half-sweep 1 from the back; a set's 2 x num_bins
 * counters are [b * 2 + half]. A vertex enters a set at most once per iteration: its stamp is the
 * tag of the last iteration whose set holds it (never cleared, so the tags of a run only grow). Its
 * bin is its current degree's (degree_bins), its half the hash of (vertex, tag) (half_sweep), so
 * adjacent vertices are separated with probability 1/2 in every iteration.
 *
 * Rule of two (ADR 0102): label_propagation is the only user; the operator stays in this folder
 * and moves to cpp/src/operators when a second algorithm uses it.
 */
#pragma once

#include "operators/execution.hpp"

#include <dyng/core/resources.hpp>

#include <cstdint>
#include <cstring>

namespace dyng::detail::lp_ops {

/**
 * @brief The half-sweep (0 or 1) of vertex v in the iteration of tag `tag` (the original's
 *        halfSweep: a hash, so adjacent vertices are separated with probability 1/2).
 * @param[in] v   The vertex.
 * @param[in] tag The iteration's tag.
 * @return 0 or 1.
 */
DYNG_HD inline int half_sweep(std::int32_t v, int tag) {
  unsigned x = static_cast<unsigned>(v) ^ (static_cast<unsigned>(tag) * 0x9E3779B9U);
  x ^= x >> 16;
  x *= 0x7FEB352DU;
  x ^= x >> 15;
  x *= 0x846CA68BU;
  x ^= x >> 16;
  return static_cast<int>(x & 1U);
}

/**
 * @brief The degree bins: bin b holds the degrees up to max_degree[b] (b < num_bins - 1), the last
 *        bin every larger degree.
 * @tparam num_bins The bins.
 */
template <int num_bins>
struct degree_bins {
  static_assert(num_bins >= 1, "a binned frontier has at least one bin");
  int max_degree[num_bins > 1 ? num_bins - 1 : 1];  ///< the bins' largest degrees, increasing

  /**
   * @brief The bin of a degree.
   * @param[in] degree A degree.
   * @return Its bin.
   */
  DYNG_HD int of(std::int64_t degree) const {
    for (int b = 0; b + 1 < num_bins; ++b) {
      if (degree <= max_degree[b]) {
        return b;
      }
    }
    return num_bins - 1;
  }
};

/**
 * @brief The offset of a half-sweep's entries in a bin's array (the original's halfOffset).
 * @param[in] half     0 or 1.
 * @param[in] capacity The bin's capacity.
 * @param[in] count    The half's entries.
 * @return 0 for half 0, capacity - count for half 1.
 */
DYNG_HD inline int half_offset(int half, int capacity, int count) {
  return half == 0 ? 0 : capacity - count;
}

/**
 * @brief The appends of one iteration into one set (trivially copyable; kernels take it by
 *        value). push() is the operator's only write.
 * @tparam id_t     Vertex id type (std::int32_t).
 * @tparam num_bins The bins.
 */
template <typename id_t, int num_bins>
struct binned_push {
  id_t* queue[num_bins];       ///< the set's arrays
  int capacity[num_bins];      ///< their capacities
  std::int32_t* counts;        ///< the set's 2 x num_bins counters
  std::int32_t* stamp;         ///< per vertex the run-wide stamps
  int tag;                     ///< the iteration's tag
  degree_bins<num_bins> bins;  ///< the degree bins

  /**
   * @brief Append v (current degree `degree`) once per iteration: on the device with one atomic
   *        per group of lanes that append to the same bin and half (the original's
   *        warp-aggregated appendToQueue), on the host in call order.
   * @param[in] v      The vertex.
   * @param[in] degree Its current degree (its bin).
   */
  DYNG_HD void push(id_t v, std::int64_t degree) const {
#if defined(__CUDA_ARCH__)
    if (stamp[v] == tag) return;
    if (atomicExch(&stamp[v], tag) == tag) return;
    const int bin = bins.of(degree);
    const int half = half_sweep(static_cast<std::int32_t>(v), tag);
    const int key = bin * 2 + half;
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, key);
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(peers) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(&counts[key], __popc(peers));
    base = __shfl_sync(peers, base, leader);
    const int idx = base + __popc(peers & ((1U << lane) - 1U));
    queue[bin][half == 0 ? idx : capacity[bin] - 1 - idx] = v;
#else
    if (stamp[v] == tag) {
      return;
    }
    stamp[v] = tag;
    const int bin = bins.of(degree);
    const int half = half_sweep(static_cast<std::int32_t>(v), tag);
    const int idx = counts[bin * 2 + half]++;
    queue[bin][half == 0 ? idx : capacity[bin] - 1 - idx] = v;
#endif
  }
};

/**
 * @brief binned_frontier<id_t, num_bins>: the two sets, their counters and the stamps (memory of
 *        the executor's backend, owned by the caller), and the counters' host mirror.
 * @tparam id_t     Vertex id type (std::int32_t).
 * @tparam num_bins The bins.
 */
template <typename id_t, int num_bins>
class binned_frontier {
 public:
  /// The memory of the frontier (the caller's; the counters 2 x 2 x num_bins words).
  struct storage {
    id_t* queue[2][num_bins];    ///< per set and bin, an array of capacity[bin]
    int capacity[num_bins];      ///< the bins' capacities
    std::int32_t* counts;        ///< 2 sets x (2 x num_bins) counters (backend memory)
    std::int32_t* host_counts;   ///< their host mirror (pinned on CUDA)
    std::int32_t* stamp;         ///< per vertex the run-wide stamps (backend memory)
    degree_bins<num_bins> bins;  ///< the degree bins
  };

  /// Counters of one set.
  static constexpr int counters_per_set = 2 * num_bins;

  /**
   * @brief A frontier over `s`.
   * @param[in] s The memory.
   */
  explicit binned_frontier(const storage& s) noexcept : s_(s) {}

  /**
   * @brief The appends of the iteration of tag `tag` into set `set`.
   * @param[in] set 0 or 1.
   * @param[in] tag The iteration's tag.
   * @return The pusher (pass it to the work by value).
   */
  [[nodiscard]] binned_push<id_t, num_bins> into(int set, int tag) const noexcept {
    binned_push<id_t, num_bins> p{};
    for (int b = 0; b < num_bins; ++b) {
      p.queue[b] = s_.queue[set][b];
      p.capacity[b] = s_.capacity[b];
    }
    p.counts = s_.counts + set * counters_per_set;
    p.stamp = s_.stamp;
    p.tag = tag;
    p.bins = s_.bins;
    return p;
  }

  /**
   * @brief Empty set `set` (its counters; asynchronous on CUDA).
   * @tparam exec_t The executor.
   * @param[in] exec The executor.
   * @param[in] set  0 or 1.
   */
  template <typename exec_t>
  void clear(const exec_t& exec, int set) const {
    exec.fill(s_.counts + set * counters_per_set, counters_per_set, std::int32_t{0});
  }

  /**
   * @brief Read set `set`'s counters into the host mirror (one host synchronization on CUDA).
   * @tparam exec_t The executor.
   * @param[in] exec The executor.
   * @param[in] set  0 or 1.
   * @return The set's size (its entries over every bin and half).
   */
  template <typename exec_t>
  std::int64_t read_size(const exec_t& exec, int set) const {
    exec.download(s_.host_counts + set * counters_per_set, s_.counts + set * counters_per_set,
                  counters_per_set);
    return size(set);
  }

  /**
   * @brief The size of set `set` from the host mirror (as read_size() last read it).
   * @param[in] set 0 or 1.
   * @return Its entries.
   */
  [[nodiscard]] std::int64_t size(int set) const noexcept {
    std::int64_t total = 0;
    for (int c = 0; c < counters_per_set; ++c) {
      total += s_.host_counts[set * counters_per_set + c];
    }
    return total;
  }

  /**
   * @brief The entries of one bin and half of set `set` (the host mirror's count).
   * @param[in] set  0 or 1.
   * @param[in] bin  The bin.
   * @param[in] half 0 or 1.
   * @return Their count.
   */
  [[nodiscard]] int count(int set, int bin, int half) const noexcept {
    return s_.host_counts[set * counters_per_set + bin * 2 + half];
  }

  /**
   * @brief Where the entries of one bin and half of set `set` start.
   * @param[in] set  0 or 1.
   * @param[in] bin  The bin.
   * @param[in] half 0 or 1.
   * @return Their first entry (backend memory).
   */
  [[nodiscard]] const id_t* list(int set, int bin, int half) const noexcept {
    return s_.queue[set][bin] + offset(bin, half, count(set, bin, half));
  }

  /**
   * @brief The offset of a half's entries in its bin's array.
   * @param[in] bin   The bin.
   * @param[in] half  0 or 1.
   * @param[in] count The half's entries.
   * @return The offset.
   */
  [[nodiscard]] int offset(int bin, int half, int count) const noexcept {
    return half_offset(half, s_.capacity[bin], count);
  }

  /// The memory.
  [[nodiscard]] const storage& memory() const noexcept {
    return s_;
  }

 private:
  storage s_;
};

/**
 * @brief The rows the frontier's expand and neighbor_reduce read: the entries [row_start[v],
 *        row_end[v]) of each vertex, an entry's column (negative: a soft-deleted entry) and
 *        weight, and the vertex states (1 = vertex_status::deleted; nullptr: every vertex active).
 * @tparam edge_t Edge offset type.
 */
template <typename edge_t>
struct slot_rows {
  const edge_t* row_start = nullptr;      ///< first entry of each row
  const edge_t* row_end = nullptr;        ///< end of each row's entries
  const std::int32_t* col_ind = nullptr;  ///< the entries' columns
  const float* weights = nullptr;         ///< their weights
  const std::int32_t* status = nullptr;   ///< the vertex states (vertex_status's values)

  /**
   * @brief The current degree of v (its entries, soft-deleted ones included: the bin rule).
   * @param[in] v A vertex.
   * @return row_end[v] - row_start[v].
   */
  DYNG_HD std::int64_t degree(std::int32_t v) const {
    return static_cast<std::int64_t>(row_end[v] - row_start[v]);
  }
};

/**
 * @brief expand on one backend (specialised per executor: the host here, cuda_exec in
 *        binned_frontier.cuh).
 * @tparam exec_t The executor.
 */
template <typename exec_t>
struct expand_impl;

/// expand on the host: the positions in order, each vertex's neighbours in row order.
template <>
struct expand_impl<operators::sequential_exec> {
  /// See expand().
  template <typename edge_t, typename id_t, int num_bins, typename eligible_t>
  static void run(const resources& /*res*/, int /*lanes*/, const slot_rows<edge_t>& rows,
                  const id_t* list, const std::uint8_t* changed, int count, eligible_t eligible,
                  const binned_push<id_t, num_bins>& out) {
    for (int i = 0; i < count; ++i) {
      if (changed[i] == 0) {
        continue;
      }
      const id_t u = list[i];
      for (edge_t e = rows.row_start[u]; e < rows.row_end[u]; ++e) {
        const std::int32_t v = rows.col_ind[e];
        if (v >= 0 && eligible(v)) {
          out.push(v, rows.degree(v));
        }
      }
      out.push(u, rows.degree(static_cast<std::int32_t>(u)));
    }
  }
};

/**
 * @brief The changed vertices of one list and their eligible neighbours into the next set
 *        (expand; the original's appendNextFrontier): for every position i with changed[i], each
 *        neighbour v of list[i] with col >= 0 and eligible(v) is pushed, and list[i] itself. On
 *        CUDA `lanes` threads share a vertex (a block of `lanes` threads above 32); the set built
 *        does not depend on the order of the pushes.
 * @tparam exec_t     The executor (operators::sequential_exec, or operators::cuda_exec with
 *                    binned_frontier.cuh).
 * @tparam edge_t     Edge offset type.
 * @tparam id_t       Vertex id type.
 * @tparam num_bins   The bins.
 * @tparam eligible_t `DYNG_HD bool operator()(std::int32_t v) const`.
 * @param[in] res      The resources (the executor's backend).
 * @param[in] lanes    Threads per vertex on CUDA (8, 32, or a block of 256 or 1024).
 * @param[in] rows     The rows.
 * @param[in] list     The list.
 * @param[in] changed  Per position whether its vertex changed.
 * @param[in] count    Its entries.
 * @param[in] eligible Which neighbours enter.
 * @param[in] out      The next set's pusher.
 */
template <typename exec_t, typename edge_t, typename id_t, int num_bins, typename eligible_t>
void expand(const resources& res, int lanes, const slot_rows<edge_t>& rows, const id_t* list,
            const std::uint8_t* changed, int count, eligible_t eligible,
            const binned_push<id_t, num_bins>& out) {
  expand_impl<exec_t>::run(res, lanes, rows, list, changed, count, eligible, out);
}

}  // namespace dyng::detail::lp_ops
