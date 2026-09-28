// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/core/parallel.hpp (worker_count, parallel_parts,
// parallel_sort)
/**
 * @file parallel_parts.hpp
 * @brief Small std::thread helpers for the readers: split [0, n) into parts, run them on threads,
 *        and a parallel sort (parts sorted in parallel, then merged pairwise in parallel rounds).
 *
 * The readers do not depend on OpenMP (a build without it reads files as fast), so they
 * parallelize with plain threads, as CycleEnumeration-GPU's parser does. Changes from the original:
 * names, an explicit thread cap (`max_threads`, 0 = the hardware concurrency, the original's only
 * choice) and no test hook for a failing thread spawner.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace dyng::detail {

/**
 * @brief Threads to use for `work` independent units of at least `grain` each.
 * @param[in] work        Amount of work (bytes, rows, ...).
 * @param[in] grain       Smallest useful amount per thread.
 * @param[in] max_threads Upper bound; 0 = std::thread::hardware_concurrency().
 * @return A thread count >= 1.
 */
inline unsigned int worker_count(std::size_t work, std::size_t grain, int max_threads = 0) {
  const unsigned int hardware = max_threads > 0 ? static_cast<unsigned int>(max_threads)
                                                : std::max(1U, std::thread::hardware_concurrency());
  const std::size_t useful = std::max<std::size_t>(1, work / std::max<std::size_t>(grain, 1));
  return static_cast<unsigned int>(std::min<std::size_t>(hardware, useful));
}

/**
 * @brief Run `body(thread, begin, end)` over `threads` contiguous parts of [0, n).
 *
 * The first exception thrown by a part (in part order) is rethrown after all parts finish. If a
 * thread cannot be started (std::system_error under a process or cgroup thread limit), the parts
 * without a thread run on the calling thread, so the work completes with fewer threads; started
 * threads are always joined before the function returns or throws.
 * @tparam body_t Callable as body(unsigned int part, std::size_t begin, std::size_t end).
 * @param[in] n       Size of the range.
 * @param[in] threads Number of parts (and threads).
 * @param[in] body    The work of one part.
 */
template <typename body_t>
void parallel_parts(std::size_t n, unsigned int threads, body_t&& body) {
  if (threads <= 1 || n == 0) {
    body(0U, std::size_t{0}, n);
    return;
  }
  std::vector<std::exception_ptr> errors(threads);
  const auto run_part = [&](unsigned int t) noexcept {
    try {
      body(t, n * t / threads, n * (t + 1) / threads);
    } catch (...) {
      errors[t] = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(threads);  // push_back below cannot throw
  for (unsigned int t = 0; t < threads; ++t) {
    try {
      pool.emplace_back([&run_part, t] { run_part(t); });
    } catch (...) {
      break;  // run this part and the remaining ones on this thread
    }
  }
  for (auto t = static_cast<unsigned int>(pool.size()); t < threads; ++t) {
    run_part(t);
  }
  for (std::thread& thread : pool) {
    thread.join();
  }
  for (const std::exception_ptr& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

/**
 * @brief Sort `values` with `less`: parts are sorted in parallel, then merged pairwise in parallel
 *        rounds.
 *
 * Not stable; equal elements may end in any order (the readers sort complete keys).
 * @tparam value_t Element type.
 * @tparam less_t  Strict weak order.
 * @param[in,out] values      The values.
 * @param[in]     less        The order.
 * @param[in]     max_threads Thread cap; 0 = the hardware concurrency.
 */
template <typename value_t, typename less_t = std::less<value_t>>
void parallel_sort(std::vector<value_t>& values, less_t less = less_t{}, int max_threads = 0) {
  const unsigned int threads = worker_count(values.size(), 1U << 16, max_threads);
  if (threads <= 1) {
    std::sort(values.begin(), values.end(), less);
    return;
  }
  std::vector<std::size_t> bounds(threads + 1);
  for (unsigned int t = 0; t <= threads; ++t) {
    bounds[t] = values.size() * t / threads;
  }
  parallel_parts(threads, threads, [&](unsigned int, std::size_t first, std::size_t last) {
    for (std::size_t part = first; part < last; ++part) {
      std::sort(values.begin() + static_cast<std::ptrdiff_t>(bounds[part]),
                values.begin() + static_cast<std::ptrdiff_t>(bounds[part + 1]), less);
    }
  });

  std::vector<value_t> buffer(values.size());
  std::vector<value_t>* source = &values;
  std::vector<value_t>* target = &buffer;
  while (bounds.size() > 2) {
    const std::size_t runs = bounds.size() - 1;
    const std::size_t pairs = (runs + 1) / 2;
    std::vector<std::size_t> next_bounds;
    next_bounds.reserve(pairs + 1);
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      next_bounds.push_back(bounds[2 * pair]);
    }
    next_bounds.push_back(bounds.back());
    parallel_parts(pairs, static_cast<unsigned int>(pairs),
                   [&](unsigned int, std::size_t first, std::size_t last) {
                     for (std::size_t pair = first; pair < last; ++pair) {
                       const auto begin = source->begin();
                       const std::size_t a = bounds[2 * pair];
                       const std::size_t b = bounds[std::min(2 * pair + 1, runs)];
                       const std::size_t c = bounds[std::min(2 * pair + 2, runs)];
                       std::merge(begin + static_cast<std::ptrdiff_t>(a),
                                  begin + static_cast<std::ptrdiff_t>(b),
                                  begin + static_cast<std::ptrdiff_t>(b),
                                  begin + static_cast<std::ptrdiff_t>(c),
                                  target->begin() + static_cast<std::ptrdiff_t>(a), less);
                     }
                   });
    std::swap(source, target);
    bounds = std::move(next_bounds);
  }
  if (source != &values) {
    values.swap(buffer);
  }
}

}  // namespace dyng::detail
