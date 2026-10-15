// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file count_delta_test.cpp
 * @brief Tests of operators::count_delta and count_owned (operators/count_delta.hpp; PLAN 4.5.3,
 *        invariants I2 and I7; ADR 0055) on the host executors and, with DYNG_TEST_CUDA=1, on
 *        cuda_exec (count_delta_cuda_test.cu supplies the device run): the sign applied once per
 *        bin, the ownership rule handed to the work, the totals independent of the thread count,
 *        the empty phase, a bin outside the range and the 64-bit limits.
 */
#include "operators/count_delta.hpp"

#include "framework/policies.hpp"
#include "operators/execution.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

namespace ops = dyng::detail::operators;
namespace fw = dyng::detail::framework;

/// Element i adds i % 7 to bin i % 3, and 1 to bin 2 when i is a multiple of 5 (a structure it
/// owns only under min_member: the rule reaches the work).
struct sample_work {
  template <typename bins_t>
  void operator()(std::int64_t i, fw::ownership::min_member rule, bins_t& bins) const {
    static_assert(std::is_same_v<decltype(rule), fw::ownership::min_member>);
    bins.add(static_cast<int>(i % 3), static_cast<std::uint64_t>(i % 7));
    if (i % 5 == 0) bins.add(2, 1);
  }
};

/// The expected totals of sample_work over [0, n).
std::array<std::uint64_t, 3> expected_totals(std::int64_t n) {
  std::array<std::uint64_t, 3> t{};
  for (std::int64_t i = 0; i < n; ++i) {
    t[static_cast<std::size_t>(i % 3)] += static_cast<std::uint64_t>(i % 7);
    if (i % 5 == 0) t[2] += 1;
  }
  return t;
}

/// OpenMP resources with `threads` threads (sequential resources in a build without OpenMP).
dyng::resources omp(int threads) {
  return dyng::backend_available(dyng::backend::openmp) ? dyng::resources::openmp(threads)
                                                        : dyng::resources::sequential();
}

template <typename exec_t>
std::vector<std::uint64_t> scratch_for(const exec_t& exec) {
  return std::vector<std::uint64_t>(ops::count_scratch_words<3>(exec), 0);
}

TEST(CountDelta, TheSignIsAppliedOncePerBinAndAccumulates) {
  const ops::sequential_exec exec(dyng::resources::sequential());
  auto scratch = scratch_for(exec);
  const std::int64_t n = 1000;
  const auto want = expected_totals(n);
  std::array<std::int64_t, 3> minus{};
  std::array<std::int64_t, 3> plus{};
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::minus, fw::ownership::min_member{}, n,
                                sample_work{}, scratch.data(), minus.data()),
            0U);
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::plus, fw::ownership::min_member{}, n, sample_work{},
                                scratch.data(), plus.data()),
            0U);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_EQ(minus[k], -static_cast<std::int64_t>(want[k])) << "bin " << k;
    EXPECT_EQ(plus[k], static_cast<std::int64_t>(want[k])) << "bin " << k;
  }
  // count_delta adds to the bins it is given (the caller keeps the phases apart or not).
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::plus, fw::ownership::min_member{}, n, sample_work{},
                                scratch.data(), minus.data()),
            0U);
  EXPECT_EQ(minus, (std::array<std::int64_t, 3>{}));
  EXPECT_EQ(ops::sign_value(fw::sign::minus), -1);
  EXPECT_EQ(ops::sign_value(fw::sign::plus), 1);
}

TEST(CountDelta, TheTotalsDoNotDependOnTheThreadCount) {
  const std::int64_t n = 100003;
  const auto want = expected_totals(n);
  for (const int threads : {1, 3, 8, 56}) {
    SCOPED_TRACE("threads " + std::to_string(threads));
    const ops::openmp_exec exec(omp(threads));
    EXPECT_GE(exec.threads(), 1);
    auto scratch = scratch_for(exec);
    std::array<std::uint64_t, 3> totals{};
    EXPECT_EQ(ops::count_owned<3>(exec, fw::ownership::min_member{}, n, sample_work{},
                                  scratch.data(), totals.data()),
              0U);
    EXPECT_EQ(totals, want);
  }
}

TEST(CountDelta, AnEmptyPhaseCountsNothing) {
  const ops::openmp_exec exec(omp(4));
  std::array<std::uint64_t, 3> totals = {7, 7, 7};
  EXPECT_EQ(ops::count_owned<3>(exec, fw::ownership::min_member{}, 0, sample_work{}, nullptr,
                                totals.data()),
            0U);
  EXPECT_EQ(totals, (std::array<std::uint64_t, 3>{}));
  std::array<std::int64_t, 3> out = {5, -5, 0};
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::minus, fw::ownership::min_member{}, 0,
                                sample_work{}, nullptr, out.data()),
            0U);
  EXPECT_EQ(out, (std::array<std::int64_t, 3>{5, -5, 0}));
}

/// Adds to bin -1 or bin 3 (outside [0, 3)) for every element divisible by 10, else to bin 0.
struct out_of_range_work {
  template <typename bins_t>
  void operator()(std::int64_t i, fw::ownership::min_member /*rule*/, bins_t& bins) const {
    bins.add(i % 10 == 0 ? (i % 20 == 0 ? -1 : 3) : 0, 1);
  }
};

TEST(CountDelta, ABinOutsideTheRangeIsReportedAndNotWritten) {
  for (const int threads : {1, 8}) {
    const ops::openmp_exec exec(omp(threads));
    auto scratch = scratch_for(exec);
    std::array<std::int64_t, 3> out{};
    const std::uint32_t status =
        ops::count_delta<3>(exec, fw::sign::plus, fw::ownership::min_member{}, 100,
                            out_of_range_work{}, scratch.data(), out.data());
    EXPECT_EQ(status & ops::count_invalid_bin, ops::count_invalid_bin);
    EXPECT_EQ(out, (std::array<std::int64_t, 3>{90, 0, 0}));
  }
}

/// Adds 2^63 to bin 0 for every element.
struct huge_work {
  template <typename bins_t>
  void operator()(std::int64_t /*i*/, fw::ownership::min_member /*rule*/, bins_t& bins) const {
    bins.add(0, std::uint64_t{1} << 63);
  }
};

TEST(CountDelta, CountsPastThe64BitRangeAreReported) {
  const ops::sequential_exec exec(dyng::resources::sequential());
  std::vector<std::uint64_t> scratch(ops::count_scratch_words<1>(exec));
  std::uint64_t total = 0;
  // Two elements: 2^64 does not fit an unsigned total.
  EXPECT_EQ(ops::count_owned<1>(exec, fw::ownership::min_member{}, 2, huge_work{}, scratch.data(),
                                &total) &
                ops::count_overflow,
            ops::count_overflow);
  // One element: 2^63 fits the total but not a signed bin; the bin is left as it was.
  std::int64_t out = 3;
  EXPECT_EQ(ops::count_delta<1>(exec, fw::sign::minus, fw::ownership::min_member{}, 1, huge_work{},
                                scratch.data(), &out),
            ops::count_overflow);
  EXPECT_EQ(out, 3);
  // A signed bin that would leave the range.
  struct big {
    void operator()(std::int64_t, fw::ownership::min_member, ops::host_count_bins<1>& b) const {
      b.add(0, 10);
    }
  };
  std::int64_t near = std::numeric_limits<std::int64_t>::max() - 5;
  EXPECT_EQ(ops::count_delta<1>(exec, fw::sign::plus, fw::ownership::min_member{}, 1, big{},
                                scratch.data(), &near),
            ops::count_overflow);
  EXPECT_EQ(near, std::numeric_limits<std::int64_t>::max() - 5);
}

TEST(CountDelta, TheScratchHoldsOneSetOfBinsPerThread) {
  EXPECT_EQ(ops::count_scratch_words<30>(ops::sequential_exec(dyng::resources::sequential())), 31U);
  const ops::openmp_exec exec(omp(6));
  EXPECT_EQ(ops::count_scratch_words<30>(exec), static_cast<std::size_t>(exec.threads()) * 31U);
}

}  // namespace
