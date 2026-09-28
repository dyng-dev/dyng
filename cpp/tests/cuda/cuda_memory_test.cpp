// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_memory_test.cpp
 * @brief The CUDA memory resources, device buffers, copies between spaces, the fill kernels and
 *        the CCCL adapters (PLAN Section 4.7.2).
 */
#include "cuda/cuda_test_kernels.hpp"
#include "support/gtest_helpers.hpp"
#include "util/device_fill.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace {

cudaMemoryType pointer_type(const void* p) {
  cudaPointerAttributes attr{};
  EXPECT_EQ(cudaPointerGetAttributes(&attr, p), cudaSuccess);
  return attr.type;
}

TEST(CudaAsyncMemoryResource, AllocatesStreamOrderedDeviceMemory) {
  DYNG_SKIP_IF_NO_CUDA();
  dyng::cuda_async_memory_resource mr(0);
  EXPECT_EQ(mr.device(), 0);
  EXPECT_EQ(mr.space(), dyng::memory_space::device);
  const dyng::stream_ref stream{};
  EXPECT_EQ(mr.used_bytes(), 0u);
  void* p = mr.allocate(stream, 1 << 20, 256);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 256, 0u);
  EXPECT_EQ(pointer_type(p), cudaMemoryTypeDevice);
  EXPECT_GE(mr.used_bytes(), std::size_t{1} << 20);
  mr.deallocate(stream, p, 1 << 20, 256);
  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  EXPECT_EQ(mr.used_bytes(), 0u);
  // Freed memory stays in the pool (release threshold unlimited): reuse needs no new reservation.
  const std::size_t reserved = mr.reserved_bytes();
  EXPECT_GE(reserved, std::size_t{1} << 20);
  for (int i = 0; i < 20; ++i) {
    void* q = mr.allocate(stream, 1 << 20, 256);
    mr.deallocate(stream, q, 1 << 20, 256);
  }
  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  EXPECT_EQ(mr.reserved_bytes(), reserved);

  EXPECT_EQ(mr.allocate(stream, 0, 8), nullptr);
  mr.deallocate(stream, nullptr, 0, 8);  // ignored
  EXPECT_THROW((void)mr.allocate(stream, 16, 512), dyng::invalid_argument_error);
  EXPECT_THROW((void)mr.allocate(stream, 16, 3), dyng::invalid_argument_error);

  void* s = mr.allocate_sync(4096, 128);
  EXPECT_EQ(pointer_type(s), cudaMemoryTypeDevice);
  mr.deallocate_sync(s, 4096, 128);
  EXPECT_EQ(mr.used_bytes(), 0u);
}

TEST(CudaApiErrors, AsyncResourceOutOfMemoryThrowsAndRecovers) {
  DYNG_SKIP_IF_NO_CUDA();
  dyng::cuda_async_memory_resource mr(0);
  const dyng::stream_ref stream{};
  constexpr std::size_t huge = std::size_t{1} << 45;  // 32 TiB
  EXPECT_THROW((void)mr.allocate(stream, huge, 256), dyng::out_of_memory_error);
  // The failure is not sticky: the runtime and the pool keep working.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  void* p = mr.allocate(stream, 1024, 256);
  EXPECT_NE(p, nullptr);
  mr.deallocate(stream, p, 1024, 256);
  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
}

TEST(CudaAsyncMemoryResource, DefaultIsOnePerDeviceAndDistinctFromUserPools) {
  DYNG_SKIP_IF_NO_CUDA();
  auto& a = dyng::default_device_memory_resource(0);
  auto& b = dyng::default_device_memory_resource(0);
  EXPECT_EQ(&a, &b);
  EXPECT_TRUE(a == b);
  dyng::cuda_async_memory_resource own(0);
  EXPECT_FALSE(own == a);
  EXPECT_NE(dyng::memory_resource_ref(own), dyng::memory_resource_ref(a));
}

TEST(PinnedHostMemoryResource, AllocatesPinnedHostMemory) {
  DYNG_SKIP_IF_NO_CUDA();
  auto& mr = dyng::default_pinned_host_memory_resource();
  EXPECT_EQ(mr.space(), dyng::memory_space::pinned_host);
  EXPECT_TRUE(mr == dyng::pinned_host_memory_resource{});
  void* p = mr.allocate(dyng::stream_ref{}, 4096, 256);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(pointer_type(p), cudaMemoryTypeHost);
  static_cast<char*>(p)[4095] = 7;  // host-accessible
  mr.deallocate(dyng::stream_ref{}, p, 4096, 256);
  EXPECT_EQ(mr.allocate_sync(0, 8), nullptr);
  EXPECT_THROW((void)mr.allocate_sync(16, 4096), dyng::invalid_argument_error);
}

TEST(CudaBuffer, DeviceBufferRoundTrip) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  std::vector<std::int64_t> host(1000);
  std::iota(host.begin(), host.end(), -500);
  auto dev = dyng::to_space(res, dyng::host_view(std::as_const(host)), dyng::memory_space::device);
  EXPECT_EQ(dev.size(), host.size());
  EXPECT_EQ(dev.space(), dyng::memory_space::device);
  EXPECT_EQ(dev.device(), 0);
  EXPECT_EQ(pointer_type(dev.data()), cudaMemoryTypeDevice);
  EXPECT_EQ(dyng::to_vector(res, dev.view()), host);

  // copy(): device -> device and device -> host.
  dyng::buffer<std::int64_t> other(res, host.size());
  dyng::copy(res, std::as_const(dev).view(), other.view());
  std::vector<std::int64_t> back(host.size(), 0);
  dyng::copy(res, std::as_const(other).view(), dyng::host_view(back));
  res.synchronize();
  EXPECT_EQ(back, host);

  // resize keeps the prefix on the device.
  other.resize(10);
  EXPECT_EQ(dyng::to_vector(res, other.view()),
            std::vector<std::int64_t>(host.begin(), host.begin() + 10));
  other.resize(2000);
  const auto grown = dyng::to_vector(res, other.view());
  EXPECT_EQ(std::vector<std::int64_t>(grown.begin(), grown.begin() + 10),
            std::vector<std::int64_t>(host.begin(), host.begin() + 10));

  // A move leaves the source empty and keeps the device memory.
  const auto* data = dev.data();
  dyng::buffer<std::int64_t> moved(std::move(dev));
  EXPECT_EQ(moved.data(), data);
  EXPECT_TRUE(dev.empty());  // NOLINT(bugprone-use-after-move): the moved-from state is tested
  EXPECT_EQ(dyng::to_vector(res, moved.view()), host);
}

TEST(CudaBuffer, KernelWritesAreVisibleThroughCopies) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  dyng::buffer<std::int64_t> b(res, 777);
  dyng::test::launch_write_sequence(res.stream(), b.data(), 777, 1000);
  const auto h = dyng::to_vector(res, b.view());
  for (std::size_t i = 0; i < h.size(); ++i) {
    ASSERT_EQ(h[i], 1000 + static_cast<std::int64_t>(i));
  }
}

TEST(CudaBuffer, PinnedStagingCopies) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  dyng::buffer<int> pinned(16, res.stream(), dyng::default_pinned_host_memory_resource());
  EXPECT_EQ(pinned.space(), dyng::memory_space::pinned_host);
  for (int i = 0; i < 16; ++i) {
    pinned[static_cast<std::size_t>(i)] = i * i;
  }
  dyng::buffer<int> dev(res, 16);
  dyng::copy(res, std::as_const(pinned).view(), dev.view());
  dyng::buffer<int> pinned_back(16, res.stream(), dyng::default_pinned_host_memory_resource());
  dyng::copy(res, std::as_const(dev).view(), pinned_back.view());
  res.synchronize();
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(pinned_back[static_cast<std::size_t>(i)], i * i);
  }
}

// to_space() takes the memory resource from the requested space (M1b review): a CUDA handle
// reaches host and pinned host memory too, and a host handle reaches pinned memory.
TEST(CudaBuffer, ToSpaceReachesEverySpaceOfTheHandle) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  const std::vector<int> src{3, 1, 4, 1, 5};
  const auto dev = dyng::to_space(res, dyng::host_view(src), dyng::memory_space::device);
  EXPECT_EQ(dev.space(), dyng::memory_space::device);
  for (const dyng::memory_space space :
       {dyng::memory_space::host, dyng::memory_space::pinned_host, dyng::memory_space::device}) {
    const auto copy = dyng::to_space(res, dev.view(), space);
    EXPECT_EQ(copy.space(), space);
    EXPECT_EQ(dyng::to_vector(res, copy.view()), src);
  }
  EXPECT_THROW((void)dyng::to_space(res, dev.view(), dyng::memory_space::managed),
               dyng::invalid_argument_error);
  const auto host = dyng::resources::sequential();
  const auto pinned = dyng::to_space(host, dyng::host_view(src), dyng::memory_space::pinned_host);
  EXPECT_EQ(pinned.space(), dyng::memory_space::pinned_host);
  EXPECT_EQ(dyng::to_vector(host, pinned.view()), src);
}

template <typename value_t>
void check_fill(const dyng::resources& res, std::size_t n, value_t value) {
  dyng::buffer<value_t> b(res, n);
  dyng::detail::fill_async(res.stream(), b.data(), n, value);
  const auto h = dyng::to_vector(res, b.view());
  ASSERT_EQ(h.size(), n);
  for (const value_t& x : h) {
    ASSERT_EQ(x, value);
  }
}

TEST(CudaFill, EveryInstantiation) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  check_fill<std::uint8_t>(res, 1, 0xAB);
  check_fill<std::int32_t>(res, 1000, -7);
  check_fill<std::uint32_t>(res, 257, 0xFFFFFFFFU);
  check_fill<std::int64_t>(res, 3'000'000, std::int64_t{1} << 40);  // several grid strides
  check_fill<std::uint64_t>(res, 5, 42);
  check_fill<float>(res, 64, 1.5F);
  check_fill<double>(res, 64, -2.25);
  dyng::detail::fill_async<std::int32_t>(res.stream(), nullptr, 0, 1);  // no-op
  res.synchronize();
}

TEST(CudaMemoryResourceRef, CcclAdaptersRoundTrip) {
  DYNG_SKIP_IF_NO_CUDA();
  if (!dyng::test::cccl3_memory_resource_available()) {
    GTEST_SKIP() << "the toolkit's CCCL predates the 3.x memory-resource concept";
  }
  const auto res = dyng::resources::cuda();
  EXPECT_TRUE(dyng::test::cccl_adapters_round_trip(res.memory(), res.stream(), 4096));
}

}  // namespace
