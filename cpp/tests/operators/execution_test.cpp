// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file execution_test.cpp
 * @brief Tests of the host executors' for_each (operators/execution.hpp; ADR 0083, the carried
 *        item (b) of 0.4): an exception thrown by the work on one element leaves
 *        openmp_exec::for_each on the calling thread after the region (through a
 *        region_exceptions carrier, ADR 0061), at 1 and 8 threads, for a throw at the first, a
 *        middle and the last element and for many throwing elements; the remaining elements stop
 *        early; an allocation failure inside the work comes out as std::bad_alloc; and
 *        sequential_exec is the plain loop it was. Its own executable: it replaces the global
 *        operator new (support/failing_allocator.hpp).
 */
#include "operators/execution.hpp"

#include "support/failing_allocator.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ops = dyng::detail::operators;

/// OpenMP resources with `threads` threads (sequential resources in a build without OpenMP).
dyng::resources omp(int threads) {
  return dyng::backend_available(dyng::backend::openmp) ? dyng::resources::openmp(threads)
                                                        : dyng::resources::sequential();
}

/// The work of one element: counts how often it ran, and throws at the marked elements.
struct counted_work {
  std::atomic<int>* runs;
  const std::vector<char>* throws_at;
  void operator()(std::int64_t i) const {
    runs[i].fetch_add(1, std::memory_order_relaxed);
    if ((*throws_at)[static_cast<std::size_t>(i)] != 0) {
      throw std::runtime_error("element " + std::to_string(i));
    }
  }
};

/// Runs for_each over n elements of which `throwing` throw; returns the message of the exception
/// that came out ("" if none) and the run counts.
template <typename exec_t>
std::string run_with_throws(const exec_t& exec, std::int64_t n,
                            const std::vector<std::int64_t>& throwing, std::vector<int>& runs_out) {
  std::unique_ptr<std::atomic<int>[]> runs(new std::atomic<int>[static_cast<std::size_t>(n)]);
  for (std::int64_t i = 0; i < n; ++i) runs[static_cast<std::size_t>(i)].store(0);
  std::vector<char> throws_at(static_cast<std::size_t>(n), 0);
  for (const std::int64_t i : throwing) throws_at[static_cast<std::size_t>(i)] = 1;
  std::string message;
  try {
    exec.for_each(n, counted_work{runs.get(), &throws_at});
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  runs_out.assign(static_cast<std::size_t>(n), 0);
  for (std::int64_t i = 0; i < n; ++i)
    runs_out[static_cast<std::size_t>(i)] = runs[static_cast<std::size_t>(i)].load();
  return message;
}

TEST(OpenmpExec, AnExceptionOfOneElementLeavesTheRegionOnTheCallingThread) {
  const std::int64_t n = 10000;
  for (const int threads : {1, 8}) {
    const ops::openmp_exec exec(omp(threads));
    for (const std::int64_t at : {std::int64_t{0}, n / 2, n - 1}) {
      SCOPED_TRACE("threads " + std::to_string(threads) + ", throw at " + std::to_string(at));
      std::vector<int> runs;
      const std::string message = run_with_throws(exec, n, {at}, runs);
      EXPECT_EQ(message, "element " + std::to_string(at));
      EXPECT_EQ(runs[static_cast<std::size_t>(at)], 1);
      std::int64_t ran = 0;
      for (std::int64_t i = 0; i < n; ++i) {
        EXPECT_LE(runs[static_cast<std::size_t>(i)], 1) << "element " << i << " ran twice";
        ran += runs[static_cast<std::size_t>(i)];
      }
      if (exec.threads() == 1) {
        // One thread takes the chunks in order: the elements before the throw ran, none after.
        for (std::int64_t i = 0; i < n; ++i)
          EXPECT_EQ(runs[static_cast<std::size_t>(i)], i <= at ? 1 : 0) << "element " << i;
      } else if (at == 0) {
        // The stop flag: once the exception is kept, the remaining elements return at once.
        EXPECT_LT(ran, n);
      }
    }
  }
}

TEST(OpenmpExec, OfManyThrowingElementsOneExceptionComesOut) {
  const std::int64_t n = 5000;
  std::vector<std::int64_t> throwing;
  for (std::int64_t i = 7; i < n; i += 3) throwing.push_back(i);
  for (const int threads : {1, 8}) {
    SCOPED_TRACE("threads " + std::to_string(threads));
    const ops::openmp_exec exec(omp(threads));
    std::vector<int> runs;
    const std::string message = run_with_throws(exec, n, throwing, runs);
    ASSERT_FALSE(message.empty());
    const std::int64_t at = std::stoll(message.substr(std::string("element ").size()));
    EXPECT_EQ((at - 7) % 3, 0) << message;
    EXPECT_EQ(runs[static_cast<std::size_t>(at)], 1);
    if (exec.threads() == 1) {
      EXPECT_EQ(at, 7);
      for (std::int64_t i = 0; i < n; ++i)
        EXPECT_EQ(runs[static_cast<std::size_t>(i)], i <= 7 ? 1 : 0) << "element " << i;
    }
  }
}

TEST(OpenmpExec, WithoutAnExceptionEveryElementRunsOnce) {
  const std::int64_t n = 100000;
  for (const int threads : {1, 8}) {
    SCOPED_TRACE("threads " + std::to_string(threads));
    const ops::openmp_exec exec(omp(threads));
    std::vector<int> runs;
    EXPECT_EQ(run_with_throws(exec, n, {}, runs), "");
    for (std::int64_t i = 0; i < n; ++i) ASSERT_EQ(runs[static_cast<std::size_t>(i)], 1) << i;
    EXPECT_EQ(run_with_throws(exec, 0, {}, runs), "");
  }
}

/// The work of one element allocates (a row of its own).
struct allocating_work {
  std::vector<std::int64_t>* rows;
  void operator()(std::int64_t i) const {
    rows[i].assign(static_cast<std::size_t>(i % 5) + 1, i);
  }
};

TEST(OpenmpExec, AnAllocationFailureInsideTheWorkIsBadAllocOnTheCallingThread) {
#if DYNG_TEST_SANITIZED
  GTEST_SKIP() << "the sanitizers replace operator new";
#endif
  const std::int64_t n = 2000;
  for (const int threads : {1, 8}) {
    SCOPED_TRACE("threads " + std::to_string(threads));
    const ops::openmp_exec exec(omp(threads));
    std::vector<std::vector<std::int64_t>> rows;
    int calls = 0;
    const int failures = dyng::test::alloc::fail_each_allocation<std::bad_alloc>(
        dyng::test::alloc::scope::any_thread,
        [&] {
          // New, empty rows (an assigned empty row would keep its capacity: no allocation).
          std::vector<std::vector<std::int64_t>>(static_cast<std::size_t>(n)).swap(rows);
          ++calls;
        },
        [&] { exec.for_each(n, allocating_work{rows.data()}); },
        [&](bool threw) {
          if (!threw) {
            for (std::int64_t i = 0; i < n; ++i)
              ASSERT_EQ(rows[static_cast<std::size_t>(i)].size(),
                        static_cast<std::size_t>(i % 5) + 1);
          }
        });
    // Every one of the n allocations of the region was failed once, and came out as bad_alloc.
    EXPECT_EQ(failures, n);
    EXPECT_EQ(calls, n + 1);
  }
}

TEST(SequentialExec, IsThePlainLoop) {
  const std::int64_t n = 1000;
  const ops::sequential_exec exec(dyng::resources::sequential());
  for (const std::int64_t at : {std::int64_t{0}, n / 2, n - 1}) {
    SCOPED_TRACE("throw at " + std::to_string(at));
    std::vector<int> runs;
    EXPECT_EQ(run_with_throws(exec, n, {at}, runs), "element " + std::to_string(at));
    for (std::int64_t i = 0; i < n; ++i)
      EXPECT_EQ(runs[static_cast<std::size_t>(i)], i <= at ? 1 : 0) << "element " << i;
  }
  std::vector<int> runs;
  EXPECT_EQ(run_with_throws(exec, n, {}, runs), "");
  for (std::int64_t i = 0; i < n; ++i) EXPECT_EQ(runs[static_cast<std::size_t>(i)], 1);
}

}  // namespace
