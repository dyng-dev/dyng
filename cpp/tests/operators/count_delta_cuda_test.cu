// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file count_delta_cuda_test.cu
 * @brief operators::count_delta on cuda_exec (operators/count_delta.hpp, ADR 0055): the same
 *        totals as the host executors, one host synchronization per non-empty count (I9), none
 *        for an empty one, and a bin outside the range reported without being written.
 */
#include "core/budget_counters.hpp"
#include "framework/policies.hpp"
#include "operators/count_delta.hpp"
#include "operators/cuda_execution.cuh"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

namespace ops = dyng::detail::operators;
namespace fw = dyng::detail::framework;

/// Element i adds i % 7 to bin i % 3, 1 to bin 2 for multiples of 5, and (when `bad`) 1 to bin 3
/// for multiples of 11.
struct device_work {
  bool bad;
  template <typename bins_t>
  __host__ __device__ void operator()(std::int64_t i, fw::ownership::min_member /*rule*/,
                                      bins_t& bins) const {
    bins.add(static_cast<int>(i % 3), static_cast<std::uint64_t>(i % 7));
    if (i % 5 == 0) bins.add(2, 1);
    if (bad && i % 11 == 0) bins.add(3, 1);
  }
};

bool have_device() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

TEST(CountDeltaCuda, TheDeviceTotalsEqualTheHostTotals) {
  if (!have_device()) GTEST_SKIP() << "no CUDA device";
  const dyng::resources res = dyng::resources::cuda();
  const ops::cuda_exec exec(res);
  dyng::buffer<std::uint64_t> scratch(res, ops::count_scratch_words<3>(exec));
  const std::int64_t n = 200003;
  std::array<std::int64_t, 3> device{};
  const std::int64_t syncs_before = dyng::detail::budget_snapshot().host_syncs;
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::minus, fw::ownership::min_member{}, n,
                                device_work{false}, scratch.data(), device.data()),
            0U);
  EXPECT_LE(dyng::detail::budget_snapshot().host_syncs - syncs_before, 1) << "one read of the bins";
  const ops::sequential_exec host(dyng::resources::sequential());
  std::array<std::uint64_t, 4> host_scratch{};
  std::array<std::int64_t, 3> expected{};
  EXPECT_EQ(ops::count_delta<3>(host, fw::sign::minus, fw::ownership::min_member{}, n,
                                device_work{false}, host_scratch.data(), expected.data()),
            0U);
  EXPECT_EQ(device, expected);
}

TEST(CountDeltaCuda, AnEmptyPhaseReadsNothingAndABadBinIsReported) {
  if (!have_device()) GTEST_SKIP() << "no CUDA device";
  const dyng::resources res = dyng::resources::cuda();
  const ops::cuda_exec exec(res);
  dyng::buffer<std::uint64_t> scratch(res, ops::count_scratch_words<3>(exec));
  std::array<std::int64_t, 3> out{};
  const std::int64_t syncs_before = dyng::detail::budget_snapshot().host_syncs;
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::plus, fw::ownership::min_member{}, 0,
                                device_work{false}, scratch.data(), out.data()),
            0U);
  EXPECT_EQ(dyng::detail::budget_snapshot().host_syncs, syncs_before) << "no read";
  EXPECT_EQ(ops::count_delta<3>(exec, fw::sign::plus, fw::ownership::min_member{}, 100,
                                device_work{true}, scratch.data(), out.data()),
            ops::count_invalid_bin);
  std::array<std::int64_t, 3> clean{};
  const ops::sequential_exec host(dyng::resources::sequential());
  std::array<std::uint64_t, 4> host_scratch{};
  EXPECT_EQ(ops::count_delta<3>(host, fw::sign::plus, fw::ownership::min_member{}, 100,
                                device_work{true}, host_scratch.data(), clean.data()),
            ops::count_invalid_bin);
  EXPECT_EQ(out, clean) << "the bad bin was not written on either side";
}

}  // namespace
