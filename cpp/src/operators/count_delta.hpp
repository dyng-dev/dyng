// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file count_delta.hpp
 * @brief count_delta: the signed count of one aggregate-delta phase, each structure counted once
 *        by the changed element that owns it, on any executor (PLAN Section 4.5.3, invariants I2
 *        and I7; ADR 0055).
 *
 * An aggregate-delta update changes a count by the structures through the changed elements:
 * minus those through the deletions on G_t, plus those through the insertions on G_{t+1}. A
 * structure through several changed elements must be counted once, by its owner (the ownership
 * rule, framework/policies.hpp). count_delta runs the work of one phase on any executor of
 * operators/execution.hpp:
 *
 *     struct owned_by_change {                    // the work of changed element i
 *       ...                                       // raw pointers into the backend's memory
 *       template <typename bins_t>
 *       DYNG_HD void operator()(std::int64_t i, framework::ownership::min_member rule,
 *                               bins_t& bins) const {
 *         // count the structures through i that `rule` gives to i ...
 *         bins.add(bin, how_many);
 *       }
 *     };
 *     std::int64_t delta[num_bins] = {};
 *     const std::uint32_t status = operators::count_delta<num_bins>(
 *         exec, framework::sign::minus, framework::ownership::min_member{}, n,
 *         owned_by_change{...}, scratch, delta);   // delta[k] -= the owned structures of bin k
 *
 * - The ownership rule is a required argument with no default (I2): the functor receives it, and
 *   a call without one, or with a type that is not an ownership rule, stops at a static_assert
 *   (cpp/tests/compile_fail/count_delta_ownership.cpp).
 * - The bins are 64-bit (I7): the functor adds unsigned counts to its bins handle; count_delta
 *   applies the sign once, to each bin's total, into signed 64-bit bins, so the subtracting and
 *   the adding phase can be kept apart by the caller. count_owned() is the same count without the
 *   sign (unsigned totals; a full count with no changed elements, such as compute()).
 * - Host executors (sequential_exec, openmp_exec): a static partition of the elements (chunks of
 *   `count_chunk`, dealt to the threads in turn), bins private to each thread (on its stack),
 *   stored in `scratch` and added up in thread order after the parallel region, so the result
 *   does not depend on the thread count or the schedule. cuda_exec: one for_each over the elements
 *   whose bins handle adds to the bins in device memory (`scratch`) with atomics, and one read of
 *   them (one host synchronization). An empty phase does no work and reads nothing.
 * - Errors are returned, not thrown, so the caller names them: count_invalid_bin when the work
 *   named a bin outside [0, num_bins) (nothing is written there), count_overflow when a total
 *   passes 2^64 - 1 (count_owned) or a signed bin leaves the 64-bit range (count_delta; the host
 *   executors only: device bins wrap).
 *
 * The rule of two (PLAN Section 4.5.3): triangle_delta (a tutorial algorithm, on every executor)
 * and triad_count's host engine (M9a, algorithms/triad_count/host_engine.hpp) count with it.
 * cycle_count keeps its own phases until its move is measured (ADR 0055).
 */
#pragma once

#include "framework/policies.hpp"
#include "operators/execution.hpp"

#include <dyng/config.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <type_traits>

#if DYNG_HAS_OPENMP && defined(_OPENMP)
#include <omp.h>
#endif

namespace dyng::detail::operators {

/// count_owned() / count_delta(): the work named a bin outside [0, num_bins) (it was not written).
inline constexpr std::uint32_t count_invalid_bin = 1U;
/// count_owned() / count_delta(): a total or a signed bin left the 64-bit range.
inline constexpr std::uint32_t count_overflow = 2U;
/// The elements of one chunk of the host executors' static partition.
inline constexpr std::int64_t count_chunk = 64;

/**
 * @brief The value of a sign.
 * @param[in] s The sign.
 * @return -1 for sign::minus, +1 for sign::plus.
 */
DYNG_HD constexpr std::int64_t sign_value(framework::sign s) noexcept {
  return s == framework::sign::minus ? -1 : 1;
}

/**
 * @brief The bins handle of count_delta's work on the host: one thread's private bins and its
 *        status word (count_invalid_bin, count_overflow).
 * @tparam num_bins The number of bins.
 */
template <int num_bins>
struct host_count_bins {
  std::uint64_t* bins;    ///< num_bins counters
  std::uint32_t* status;  ///< the error bits of this thread

  /**
   * @brief Add `value` to bin `bin`; a bin outside [0, num_bins) sets count_invalid_bin instead.
   * @param[in] bin   The bin.
   * @param[in] value The count.
   */
  DYNG_HD void add(int bin, std::uint64_t value) const {
    if (bin < 0 || bin >= num_bins) {
      *status |= count_invalid_bin;
      return;
    }
#if defined(__CUDA_ARCH__)
    bins[bin] += value;
#else
    if (__builtin_add_overflow(bins[bin], value, &bins[bin])) {
      *status |= count_overflow;
    }
#endif
  }
};

/**
 * @brief The bins handle of count_delta's work on the device: atomic additions to the bins in
 *        device memory; a bin outside [0, num_bins) counts in the word after them instead.
 * @tparam num_bins The number of bins.
 */
template <int num_bins>
struct device_count_bins {
  std::uint64_t* bins;  ///< num_bins counters, then the invalid-bin word

  /**
   * @brief Add `value` to bin `bin` (atomically).
   * @param[in] bin   The bin.
   * @param[in] value The count.
   */
  DYNG_HD void add(int bin, std::uint64_t value) const {
    if (bin < 0 || bin >= num_bins) {
      atomic_add(bins + num_bins, std::uint64_t{1});
      return;
    }
    atomic_add(bins + bin, value);
  }
};

namespace count_detail {

/// A type that is never true (a static_assert that fires only when its template is instantiated).
template <typename>
inline constexpr bool always_false = false;

/// The element of cuda_exec's for_each: the work of element i with the device bins handle.
template <int num_bins, typename ownership_t, typename function_t>
struct device_item {
  function_t f;         ///< the work
  ownership_t rule;     ///< the ownership rule
  std::uint64_t* bins;  ///< num_bins + 1 words of device memory

  /// Element i.
  DYNG_HD void operator()(std::int64_t i) const {
    device_count_bins<num_bins> handle{bins};
    f(i, rule, handle);
  }
};

/// The host count: the static partition, private bins per thread, the totals in thread order.
template <int num_bins, typename ownership_t, typename function_t>
std::uint32_t host_count(int threads, ownership_t rule, std::int64_t n, const function_t& f,
                         std::uint64_t* scratch, std::uint64_t* totals) {
  constexpr std::size_t stride = static_cast<std::size_t>(num_bins) + 1;
  for (std::size_t w = 0; w < static_cast<std::size_t>(threads) * stride; ++w) {
    scratch[w] = 0;
  }
  // The work of thread t: the chunks t, t + threads, ... of the elements (a static partition).
  auto run = [&](int t, int team) {
    std::uint64_t mine[num_bins] = {};
    std::uint32_t status = 0;
    host_count_bins<num_bins> handle{mine, &status};
    const std::int64_t chunks = (n + count_chunk - 1) / count_chunk;
    for (std::int64_t c = t; c < chunks; c += team) {
      const std::int64_t end =
          c * count_chunk + count_chunk < n ? c * count_chunk + count_chunk : n;
      for (std::int64_t i = c * count_chunk; i < end; ++i) {
        f(i, rule, handle);
      }
    }
    std::uint64_t* slot = scratch + static_cast<std::size_t>(t) * stride;
    for (int k = 0; k < num_bins; ++k) {
      slot[k] = mine[k];
    }
    slot[num_bins] = status;
  };
#if DYNG_HAS_OPENMP && defined(_OPENMP)
  if (threads > 1) {
    std::exception_ptr failure;
#pragma omp parallel num_threads(threads)
    {
      try {
        run(omp_get_thread_num(), omp_get_num_threads());
      } catch (...) {
#pragma omp critical(dyng_operators_count_delta)
        if (!failure) {
          failure = std::current_exception();
        }
      }
    }
    if (failure) {
      std::rethrow_exception(failure);
    }
  } else {
    run(0, 1);
  }
#else
  (void)threads;
  run(0, 1);
#endif
  std::uint32_t status = 0;
  for (int k = 0; k < num_bins; ++k) {
    totals[k] = 0;
  }
  for (int t = 0; t < threads; ++t) {  // thread order
    const std::uint64_t* slot = scratch + static_cast<std::size_t>(t) * stride;
    for (int k = 0; k < num_bins; ++k) {
      if (__builtin_add_overflow(totals[k], slot[k], &totals[k])) {
        status |= count_overflow;
      }
    }
    status |= static_cast<std::uint32_t>(slot[num_bins]);
  }
  return status;
}

}  // namespace count_detail

/**
 * @brief The words of scratch count_owned() and count_delta() need on an executor (in the
 *        executor's memory): one set of bins and a status word per host thread; on the device the
 *        bins and the invalid-bin word.
 * @tparam num_bins The number of bins.
 * @tparam exec_t   The executor.
 * @param[in] exec The executor.
 * @return The number of 64-bit words.
 */
template <int num_bins, typename exec_t>
[[nodiscard]] std::size_t count_scratch_words(const exec_t& exec) noexcept {
  static_assert(num_bins >= 1, "count_delta: num_bins must be at least 1");
  if constexpr (exec_t::on_device) {
    (void)exec;
    return static_cast<std::size_t>(num_bins) + 1;
  } else {
    return static_cast<std::size_t>(exec.threads()) * (static_cast<std::size_t>(num_bins) + 1);
  }
}

/**
 * @brief The unsigned count of one phase: the totals per bin of the structures `f` counts for the
 *        elements [0, n), each element counting the structures `rule` gives it (see the file
 *        comment).
 * @tparam num_bins    The number of bins (at least 1).
 * @tparam exec_t      The executor (operators::sequential_exec, openmp_exec or cuda_exec).
 * @tparam ownership_t The ownership rule (framework::ownership); required, no default (I2).
 * @tparam function_t  The work of one element: `DYNG_HD void operator()(std::int64_t i,
 *                     ownership_t rule, bins_t& bins) const`, adding with `bins.add(bin, count)`;
 *                     it must not throw. Trivially copyable on cuda_exec.
 * @param[in]  exec    The executor.
 * @param[in]  rule    The ownership rule the work applies.
 * @param[in]  n       Number of elements.
 * @param[in]  f       The work.
 * @param[out] scratch count_scratch_words<num_bins>(exec) words of the executor's memory.
 * @param[out] totals  num_bins totals (host memory).
 * @return 0, or the error bits count_invalid_bin and count_overflow.
 * @sync On cuda_exec one host synchronization when n > 0 (the read of the bins).
 */
template <int num_bins, typename exec_t, typename ownership_t, typename function_t>
[[nodiscard]] std::uint32_t count_owned(const exec_t& exec, ownership_t rule, std::int64_t n,
                                        const function_t& f, std::uint64_t* scratch,
                                        std::uint64_t* totals) {
  static_assert(framework::is_ownership_policy_v<ownership_t>,
                "count_delta: the third argument must be an ownership rule "
                "(framework::ownership::min_member): every structure through several changed "
                "elements is counted once, by its owner (invariant I2)");
  static_assert(num_bins >= 1, "count_delta: num_bins must be at least 1");
  if (n <= 0) {
    for (int k = 0; k < num_bins; ++k) {
      totals[k] = 0;
    }
    return 0;  // no work, no read
  }
  if constexpr (exec_t::on_device) {
    exec.fill(scratch, static_cast<std::int64_t>(num_bins) + 1, std::uint64_t{0});
    exec.for_each(n,
                  count_detail::device_item<num_bins, ownership_t, function_t>{f, rule, scratch});
    std::uint64_t host[num_bins + 1] = {};
    exec.download(host, scratch, static_cast<std::int64_t>(num_bins) + 1);
    for (int k = 0; k < num_bins; ++k) {
      totals[k] = host[k];
    }
    return host[num_bins] != 0 ? count_invalid_bin : 0U;
  } else {
    return count_detail::host_count<num_bins>(exec.threads(), rule, n, f, scratch, totals);
  }
}

/**
 * @brief Without an ownership rule: a compile error (I2: exactly-once counting has no default
 *        rule).
 * @tparam num_bins   The number of bins.
 * @tparam exec_t     The executor.
 * @tparam function_t The work.
 * @return Nothing (never compiles).
 */
template <int num_bins, typename exec_t, typename function_t>
std::uint32_t count_owned(const exec_t&, std::int64_t, const function_t&, std::uint64_t*,
                          std::uint64_t*) {
  static_assert(count_detail::always_false<exec_t>,
                "count_delta needs an ownership rule: pass framework::ownership::min_member{} "
                "after the sign (invariant I2: exactly-once counting has no default rule)");
  return 0;
}

/**
 * @brief The signed count of one aggregate-delta phase: `out[k] += sign * (the structures of bin
 *        k that the elements [0, n) own)` (see the file comment and count_owned()).
 * @tparam num_bins    The number of bins (at least 1).
 * @tparam exec_t      The executor.
 * @tparam ownership_t The ownership rule (framework::ownership); required, no default (I2).
 * @tparam function_t  The work of one element (count_owned()).
 * @param[in]     exec    The executor.
 * @param[in]     s       The sign of the phase (minus: the deletions on G_t; plus: the
 *                        insertions on G_{t+1}).
 * @param[in]     rule    The ownership rule the work applies.
 * @param[in]     n       Number of elements (the changed elements of the phase).
 * @param[in]     f       The work.
 * @param[out]    scratch count_scratch_words<num_bins>(exec) words of the executor's memory.
 * @param[in,out] out     num_bins signed bins (host memory), added to.
 * @return 0, or the error bits count_invalid_bin and count_overflow (a bin that would leave the
 *         64-bit range is not changed).
 * @sync On cuda_exec one host synchronization when n > 0.
 */
template <int num_bins, typename exec_t, typename ownership_t, typename function_t>
[[nodiscard]] std::uint32_t count_delta(const exec_t& exec, framework::sign s, ownership_t rule,
                                        std::int64_t n, const function_t& f, std::uint64_t* scratch,
                                        std::int64_t* out) {
  std::uint64_t totals[num_bins] = {};
  std::uint32_t status = count_owned<num_bins>(exec, rule, n, f, scratch, totals);
  constexpr auto largest = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  for (int k = 0; k < num_bins; ++k) {
    if (totals[k] > largest) {
      status |= count_overflow;
      continue;
    }
    const std::int64_t value = sign_value(s) * static_cast<std::int64_t>(totals[k]);
    std::int64_t sum = 0;
    if (__builtin_add_overflow(out[k], value, &sum)) {
      status |= count_overflow;
      continue;
    }
    out[k] = sum;
  }
  return status;
}

/**
 * @brief Without an ownership rule: a compile error (I2: exactly-once counting has no default
 *        rule).
 * @tparam num_bins   The number of bins.
 * @tparam exec_t     The executor.
 * @tparam function_t The work.
 * @return Nothing (never compiles).
 */
template <int num_bins, typename exec_t, typename function_t>
std::uint32_t count_delta(const exec_t&, framework::sign, std::int64_t, const function_t&,
                          std::uint64_t*, std::int64_t*) {
  static_assert(count_detail::always_false<exec_t>,
                "count_delta needs an ownership rule: pass framework::ownership::min_member{} "
                "after the sign (invariant I2: exactly-once counting has no default rule)");
  return 0;
}

}  // namespace dyng::detail::operators
