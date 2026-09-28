// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:headers/listGather.h (ListGather)
/**
 * @file list_gather.hpp
 * @brief list_gather: concatenate per-thread lists of an OpenMP parallel region into a shared
 *        vector at prefix-sum offsets.
 *
 * Mechanical changes from ListGather: templated on the element type, and the offsets are sized
 * from the thread count the caller passes to `num_threads(...)` (resources::num_threads()),
 * instead of omp_get_max_threads(), because dynG never changes the global OpenMP setting and a
 * region may request more threads than that setting; gather() accepts lists with any allocator
 * (the engines pass cache-line padded thread_lists kept in their workspace, util/thread_list.hpp).
 *
 * Added: gather_pair(), which gathers two lists of a region with the barriers of one gather.
 * ListGather's gather() costs two barriers (before and after the prefix sum), so a region that
 * gathers two lists after an `omp for` passed six barriers (the loop's, two per gather, the
 * region's); with `omp for nowait` and gather_pair() it passes three. The lists, their order and
 * the outputs are the same. The near-far loop of the sssp engines runs about a thousand such
 * regions per objective on a 10K batch, where the barriers are most of its time (M1b,
 * parity/results/M1b.md section 9).
 */
#pragma once

#include <dyng/config.hpp>

#if DYNG_HAS_OPENMP

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dyng::detail {

/**
 * @brief Concatenate per-thread lists into a shared vector.
 *
 * Every thread of a parallel region calls gather() exactly once with its local list; the lists
 * are appended to the output (after its current contents) in thread order, at offsets from a
 * prefix sum over their sizes, so the copies run in parallel instead of being serialized in a
 * critical section. gather() contains barriers.
 *
 * @tparam value_t Element type.
 */
template <typename value_t>
class list_gather {
 public:
  /**
   * @brief Prepare a gather into `out`.
   * @param[in,out] out         The shared output (appended to).
   * @param[in]     max_threads The thread count requested for the region (>= its team size).
   */
  list_gather(std::vector<value_t>& out, int max_threads)
      : out_(out), base_(out.size()), offsets_(static_cast<std::size_t>(max_threads) + 1, 0) {}

  /**
   * @brief Append this thread's list (call once per thread, inside the parallel region).
   * @tparam alloc_t Allocator of the local list (std::allocator or cache_line_allocator).
   * @param[in] local The thread's list.
   */
  template <typename alloc_t>
  void gather(const std::vector<value_t, alloc_t>& local) {
    const int thread = omp_get_thread_num();
    offsets_[static_cast<std::size_t>(thread) + 1] = local.size();
#pragma omp barrier
#pragma omp single
    {
      const int threads = omp_get_num_threads();
      for (int i = 0; i < threads; ++i) {
        offsets_[static_cast<std::size_t>(i) + 1] += offsets_[static_cast<std::size_t>(i)];
      }
      out_.resize(base_ + offsets_[static_cast<std::size_t>(threads)]);
    }
    std::copy(local.begin(), local.end(),
              out_.begin() +
                  static_cast<std::ptrdiff_t>(base_ + offsets_[static_cast<std::size_t>(thread)]));
  }

  /**
   * @brief Append this thread's lists to two gathers at once (call once per thread, inside the
   *        parallel region, instead of first.gather(...) followed by second.gather(...)).
   *
   * The same result as the two gather() calls, with one barrier before and one after the prefix
   * sums instead of two each. The region's work-sharing loop may use `nowait`: the first barrier
   * waits for every thread's part of it.
   *
   * @tparam first_alloc_t  Allocator of the first local list.
   * @tparam second_alloc_t Allocator of the second local list.
   * @param[in,out] first        The first gather.
   * @param[in]     local_first  The thread's list for the first gather.
   * @param[in,out] second       The second gather (a different output than the first's).
   * @param[in]     local_second The thread's list for the second gather.
   */
  template <typename first_alloc_t, typename second_alloc_t>
  static void gather_pair(list_gather& first,
                          const std::vector<value_t, first_alloc_t>& local_first,
                          list_gather& second,
                          const std::vector<value_t, second_alloc_t>& local_second) {
    const auto thread = static_cast<std::size_t>(omp_get_thread_num());
    first.offsets_[thread + 1] = local_first.size();
    second.offsets_[thread + 1] = local_second.size();
#pragma omp barrier
#pragma omp single
    {
      const auto threads = static_cast<std::size_t>(omp_get_num_threads());
      first.prefix_sum_and_resize(threads);
      second.prefix_sum_and_resize(threads);
    }
    first.copy_at(local_first, thread);
    second.copy_at(local_second, thread);
  }

 private:
  void prefix_sum_and_resize(std::size_t threads) {
    for (std::size_t i = 0; i < threads; ++i) {
      offsets_[i + 1] += offsets_[i];
    }
    out_.resize(base_ + offsets_[threads]);
  }

  template <typename alloc_t>
  void copy_at(const std::vector<value_t, alloc_t>& local, std::size_t thread) {
    std::copy(local.begin(), local.end(),
              out_.begin() + static_cast<std::ptrdiff_t>(base_ + offsets_[thread]));
  }

  std::vector<value_t>& out_;
  std::size_t base_;
  std::vector<std::size_t> offsets_;
};

}  // namespace dyng::detail

#endif  // DYNG_HAS_OPENMP
