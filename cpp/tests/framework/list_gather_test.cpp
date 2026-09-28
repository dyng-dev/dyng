// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file list_gather_test.cpp
 * @brief list_gather: gather() and gather_pair() concatenate per-thread lists in thread order
 *        (util/list_gather.hpp).
 */
#include <dyng/config.hpp>

#include <gtest/gtest.h>

#if DYNG_HAS_OPENMP

#include "util/list_gather.hpp"
#include "util/thread_list.hpp"

#include <omp.h>

#include <cstdint>
#include <vector>

namespace {

using dyng::detail::list_gather;
using dyng::detail::thread_list;

// Thread t contributes t + 1 elements to the first list and 2 * t to the second, each tagged
// with the thread number, after an `omp for nowait` that fills a shared array.
struct gathered {
  std::vector<std::int32_t> first;
  std::vector<std::int32_t> second;
  std::vector<std::int32_t> filled;
  int team = 0;
};

gathered run(int threads, bool paired) {
  gathered out;
  out.first = {-1, -2};  // appended after the existing contents
  out.filled.assign(1000, 0);
  list_gather<std::int32_t> first(out.first, threads);
  list_gather<std::int32_t> second(out.second, threads);
#pragma omp parallel num_threads(threads)
  {
    const int t = omp_get_thread_num();
#pragma omp single
    out.team = omp_get_num_threads();
    thread_list<std::int32_t> local_first(static_cast<std::size_t>(t) + 1, t);
    std::vector<std::int32_t> local_second(2 * static_cast<std::size_t>(t), 100 + t);
#pragma omp for schedule(dynamic, 7) nowait
    for (int i = 0; i < 1000; ++i) {
      out.filled[static_cast<std::size_t>(i)] = i;
    }
    if (paired) {
      list_gather<std::int32_t>::gather_pair(first, local_first, second, local_second);
    } else {
      first.gather(local_first);
      second.gather(local_second);
    }
    // The first barrier of a gather waited for every thread's part of the loop.
    for (int i = 0; i < 1000; ++i) {
      EXPECT_EQ(out.filled[static_cast<std::size_t>(i)], i);
    }
  }
  return out;
}

TEST(ListGather, PairEqualsTwoGathersInThreadOrder) {
  for (int threads : {1, 3, 8}) {
    const gathered two = run(threads, false);
    const gathered pair = run(threads, true);
    std::vector<std::int32_t> first = {-1, -2};
    std::vector<std::int32_t> second;
    for (int t = 0; t < two.team; ++t) {
      first.insert(first.end(), static_cast<std::size_t>(t) + 1, t);
      second.insert(second.end(), 2 * static_cast<std::size_t>(t), 100 + t);
    }
    EXPECT_EQ(two.first, first) << threads;
    EXPECT_EQ(two.second, second) << threads;
    EXPECT_EQ(pair.first, first) << threads;
    EXPECT_EQ(pair.second, second) << threads;
  }
}

}  // namespace

#endif  // DYNG_HAS_OPENMP
