// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_resources_test.cpp
 * @brief resources::cuda(): availability, the recorded device capabilities, streams, the memory
 *        resource rules, warm_up() and the device guard (PLAN Sections 4.6 and 4.7.1).
 */
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "support/gtest_helpers.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace {

using dyng::detail::resources_access;

TEST(CudaBackend, AvailableAndPreferred) {
  DYNG_SKIP_IF_NO_CUDA();
  EXPECT_GT(dyng::detail::cuda_device_count(), 0);
  EXPECT_EQ(dyng::default_backend(), dyng::backend::cuda);
  const dyng::resources res;  // the default handle is cuda when a device is visible
  EXPECT_EQ(res.get_backend(), dyng::backend::cuda);
}

TEST(CudaResources, Defaults) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  EXPECT_EQ(res.get_backend(), dyng::backend::cuda);
  EXPECT_EQ(res.device(), 0);
  EXPECT_EQ(res.num_threads(), 1);
  EXPECT_EQ(res.default_space(), dyng::memory_space::device);
  EXPECT_EQ(res.memory().space(), dyng::memory_space::device);
  EXPECT_EQ(res.memory(), dyng::memory_resource_ref(dyng::default_device_memory_resource(0)));
  EXPECT_EQ(resources_access::staging_memory(res).space(), dyng::memory_space::pinned_host);
  EXPECT_TRUE(res.stream().is_per_thread_default());
  EXPECT_NO_THROW(res.synchronize());
}

TEST(CudaResources, RecordsTheDeviceOnce) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda(0);
  const auto& props = resources_access::device_properties(res);
  EXPECT_EQ(props.ordinal, 0);
  EXPECT_FALSE(props.name.empty());
  EXPECT_GT(props.multiprocessor_count, 0);
  EXPECT_GE(props.major, 7);  // sm_75 is the floor of the release list
  EXPECT_GT(props.max_threads_per_multiprocessor, 0);
  EXPECT_GT(props.total_global_memory, 0u);
  EXPECT_TRUE(props.memory_pools);
  int coop = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0), cudaSuccess);
  EXPECT_EQ(props.cooperative_launch, coop != 0);

  // The forced flag (for the engine-selection tests) affects every copy of the handle only.
  const auto copy = res;  // NOLINT(performance-unnecessary-copy-initialization)
  resources_access::force_cooperative_launch(res, false);
  EXPECT_FALSE(resources_access::device_properties(copy).cooperative_launch);
  const auto other = dyng::resources::cuda(0);
  EXPECT_EQ(resources_access::device_properties(other).cooperative_launch, coop != 0);
  resources_access::force_cooperative_launch(res, true);
  EXPECT_TRUE(resources_access::device_properties(copy).cooperative_launch);

  // Host handles have no device record.
  EXPECT_THROW((void)resources_access::device_properties(dyng::resources::sequential()),
               dyng::internal_error);
}

TEST(CudaResources, InvalidDevice) {
  DYNG_SKIP_IF_NO_CUDA();
  const int count = dyng::detail::cuda_device_count();
  EXPECT_THROW((void)dyng::resources::cuda(-1), dyng::invalid_argument_error);
  try {
    (void)dyng::resources::cuda(count);
    ADD_FAILURE() << "expected invalid_argument_error";
  } catch (const dyng::invalid_argument_error& e) {
    EXPECT_NE(std::string(e.what()).find("does not exist"), std::string::npos) << e.what();
  }
  EXPECT_THROW((void)dyng::default_device_memory_resource(count), dyng::invalid_argument_error);
}

TEST(CudaResources, UserStream) {
  DYNG_SKIP_IF_NO_CUDA();
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  {
    const auto res = dyng::resources::cuda(0, dyng::stream_ref(stream));
    EXPECT_EQ(res.stream().get(), stream);
    EXPECT_FALSE(res.stream().is_per_thread_default());
    void* p = res.memory().allocate(res.stream(), 1024, 256);
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(cudaMemsetAsync(p, 0, 1024, stream), cudaSuccess);
    res.memory().deallocate(res.stream(), p, 1024, 256);
    EXPECT_NO_THROW(res.synchronize());
    EXPECT_NO_THROW(res.stream().synchronize());
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

/// A device-space resource wrapping another one (to check set_memory_resource()).
class forwarding_resource {
 public:
  explicit forwarding_resource(dyng::memory_resource_ref upstream) : upstream_(upstream) {}
  void* allocate(dyng::stream_ref s, std::size_t b, std::size_t a) {
    return upstream_.allocate(s, b, a);
  }
  void deallocate(dyng::stream_ref s, void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate(s, p, b, a);
  }
  void* allocate_sync(std::size_t b, std::size_t a) {
    return upstream_.allocate_sync(b, a);
  }
  void deallocate_sync(void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate_sync(p, b, a);
  }
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return upstream_.space();
  }

 private:
  dyng::memory_resource_ref upstream_;
};

TEST(CudaResources, MemoryResourceMustBeDeviceAccessible) {
  DYNG_SKIP_IF_NO_CUDA();
  auto res = dyng::resources::cuda();
  EXPECT_THROW(res.set_memory_resource(dyng::default_host_memory_resource()),
               dyng::invalid_argument_error);
  EXPECT_THROW(res.set_memory_resource(dyng::default_pinned_host_memory_resource()),
               dyng::invalid_argument_error);
  forwarding_resource device_mr(dyng::default_device_memory_resource(0));
  EXPECT_NO_THROW(res.set_memory_resource(device_mr));
  EXPECT_EQ(res.memory(), dyng::memory_resource_ref(device_mr));
  // A host backend still refuses device memory.
  auto host = dyng::resources::sequential();
  EXPECT_THROW(host.set_memory_resource(device_mr), dyng::invalid_argument_error);
}

TEST(CudaResources, WarmUpLoadsEveryRegisteredKernel) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  const auto kernels = dyng::detail::registered_kernels();
  // The library registers at least its fill kernels (7 instantiations).
  EXPECT_GE(kernels.size(), 7u);
  bool has_fill = false;
  for (const auto& k : kernels) {
    ASSERT_NE(k.function, nullptr);
    has_fill = has_fill || std::strstr(k.name, "fill_kernel") != nullptr;
  }
  EXPECT_TRUE(has_fill);
  EXPECT_EQ(dyng::detail::cuda_warm_up(0, res.stream(), res.memory()), kernels.size());
  EXPECT_NO_THROW(res.warm_up());
  EXPECT_NO_THROW(dyng::resources::sequential().warm_up());
}

TEST(CudaResources, CallsDoNotChangeTheCurrentDevice) {
  DYNG_SKIP_IF_NO_CUDA();
  if (dyng::detail::cuda_device_count() < 2) {
    GTEST_SKIP() << "needs two visible devices";
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  const auto res = dyng::resources::cuda(0);
  res.warm_up();
  void* p = res.memory().allocate(res.stream(), 64, 64);
  cudaPointerAttributes attr{};
  ASSERT_EQ(cudaPointerGetAttributes(&attr, p), cudaSuccess);
  EXPECT_EQ(attr.device, 0);
  res.memory().deallocate(res.stream(), p, 64, 64);
  // Copies are ordered on the handle's per-thread stream of device 0, not of the current device.
  const std::vector<int> host{1, 2, 3, 4};
  auto dev = dyng::to_space(res, dyng::host_view(host), dyng::memory_space::device);
  EXPECT_EQ(dyng::to_vector(res, dev.view()), host);
  dev.resize(8);
  EXPECT_EQ(dyng::to_vector(res, dev.view())[3], 4);
  res.synchronize();
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

TEST(CudaResources, DeviceGuardRestores) {
  DYNG_SKIP_IF_NO_CUDA();
  int before = -1;
  ASSERT_EQ(cudaGetDevice(&before), cudaSuccess);
  {
    const dyng::detail::scoped_device guard(dyng::detail::cuda_device_count() - 1);
    int inside = -1;
    ASSERT_EQ(cudaGetDevice(&inside), cudaSuccess);
    EXPECT_EQ(inside, dyng::detail::cuda_device_count() - 1);
  }
  int after = -1;
  ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
  EXPECT_EQ(after, before);
}

}  // namespace
