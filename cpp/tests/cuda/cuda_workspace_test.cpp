// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_workspace_test.cpp
 * @brief The workspace pool of a CUDA handle (ADR 0015 on the device): device scratch arrays
 *        allocated once and reused, no allocation in steady state (invariant I9), release through
 *        the handle's memory resource and stream, and stream order across the per-thread default
 *        streams of two threads (the lease fences, M1b review).
 */
#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <thread>
#include <vector>

namespace {

using dyng::detail::pooled_workspace;
using dyng::detail::resources_access;
using dyng::detail::scratch_buffer;

struct device_scratch final : pooled_workspace {
  scratch_buffer<std::int32_t> ids;
  scratch_buffer<std::int64_t> distances;
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return ids.bytes() + distances.bytes();
  }
};

/// Counts the calls into an upstream device resource.
class counting_resource {
 public:
  explicit counting_resource(dyng::memory_resource_ref upstream) : upstream_(upstream) {}
  void* allocate(dyng::stream_ref s, std::size_t b, std::size_t a) {
    ++allocations;
    return upstream_.allocate(s, b, a);
  }
  void deallocate(dyng::stream_ref s, void* p, std::size_t b, std::size_t a) noexcept {
    ++deallocations;
    upstream_.deallocate(s, p, b, a);
  }
  void* allocate_sync(std::size_t b, std::size_t a) {
    ++allocations;
    return upstream_.allocate_sync(b, a);
  }
  void deallocate_sync(void* p, std::size_t b, std::size_t a) noexcept {
    ++deallocations;
    upstream_.deallocate_sync(p, b, a);
  }
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return upstream_.space();
  }
  std::atomic<int> allocations{0};
  std::atomic<int> deallocations{0};

 private:
  dyng::memory_resource_ref upstream_;
};

cudaMemoryType pointer_type(const void* p) {
  cudaPointerAttributes attr{};
  EXPECT_EQ(cudaPointerGetAttributes(&attr, p), cudaSuccess);
  return attr.type;
}

void run(const dyng::resources& res, std::size_t n) {
  auto ws = resources_access::workspaces(res).acquire<device_scratch>();
  std::int32_t* ids = ws->ids.reserve(res, n);
  std::int64_t* dist = ws->distances.reserve(res, n);
  ASSERT_EQ(cudaMemsetAsync(ids, 0, n * sizeof(std::int32_t),
                            static_cast<cudaStream_t>(res.stream().get())),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(dist, 0, n * sizeof(std::int64_t),
                            static_cast<cudaStream_t>(res.stream().get())),
            cudaSuccess);
}

TEST(CudaWorkspace, DeviceScratchIsReusedAndSteadyStateAllocatesNothing) {
  DYNG_SKIP_IF_NO_CUDA();
  dyng::cuda_async_memory_resource pool(0);
  counting_resource counting(pool);
  auto res = dyng::resources::cuda();
  res.set_memory_resource(counting);

  const std::int32_t* first = nullptr;
  {
    auto ws = resources_access::workspaces(res).acquire<device_scratch>();
    first = ws->ids.reserve(res, 100'000);
    ws->distances.reserve(res, 100'000);
    EXPECT_EQ(pointer_type(first), cudaMemoryTypeDevice);
    EXPECT_EQ(ws->ids.space(), dyng::memory_space::device);
  }
  EXPECT_EQ(counting.allocations.load(), 2);
  EXPECT_EQ(res.workspace_bytes(), 100'000 * (sizeof(std::int32_t) + sizeof(std::int64_t)));

  // Steady state: the same (or a smaller) size reuses the arrays, with no allocation at all.
  for (int i = 0; i < 50; ++i) {
    run(res, i % 2 == 0 ? 100'000 : 60'000);
  }
  EXPECT_EQ(counting.allocations.load(), 2);
  EXPECT_EQ(counting.deallocations.load(), 0);
  {
    auto ws = resources_access::workspaces(res).acquire<device_scratch>();
    EXPECT_EQ(ws->ids.data(), first);
  }
  const auto stats = resources_access::workspaces(res).statistics();
  EXPECT_EQ(stats.created, 1u);
  res.synchronize();
  EXPECT_GE(pool.used_bytes(), res.workspace_bytes());

  // Growing reallocates once (the old array first).
  run(res, 200'000);
  EXPECT_EQ(counting.allocations.load(), 4);
  EXPECT_EQ(counting.deallocations.load(), 2);

  // Releasing frees through the handle's resource and stream.
  res.release_workspaces();
  EXPECT_EQ(counting.deallocations.load(), 4);
  EXPECT_EQ(res.workspace_bytes(), 0u);
  res.synchronize();
  EXPECT_EQ(pool.used_bytes(), 0u);
}

TEST(CudaWorkspace, SetMemoryResourceAndHandleDestructionRelease) {
  DYNG_SKIP_IF_NO_CUDA();
  dyng::cuda_async_memory_resource pool(0);
  counting_resource counting(pool);
  {
    auto res = dyng::resources::cuda();
    res.set_memory_resource(counting);
    run(res, 1000);
    EXPECT_EQ(counting.allocations.load(), 2);
    // Switching resources releases the idle workspaces allocated from the previous one.
    res.set_memory_resource(dyng::default_device_memory_resource(0));
    EXPECT_EQ(counting.deallocations.load(), 2);
    res.set_memory_resource(counting);
    run(res, 1000);
    EXPECT_EQ(counting.allocations.load(), 4);
    res.synchronize();
  }
  // The last copy of the handle frees its workspaces.
  EXPECT_EQ(counting.deallocations.load(), 4);
  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  EXPECT_EQ(pool.used_bytes(), 0u);
}

/// Occupies the stream it is enqueued on for a while (a host function, so no kernel is needed).
void hold_the_stream(void* /*unused*/) {
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

// Copies of one default-stream handle used on two threads run on two streams: cudaStreamPerThread
// is a different stream on every thread. A workspace that thread A returns while its device work
// is still pending must not be reused by thread B before that work has completed.
TEST(CudaWorkspace, ALeaseOnAnotherThreadsDefaultStreamWaitsForThePreviousLease) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  ASSERT_TRUE(res.stream().is_per_thread_default());
  constexpr std::size_t n = 1 << 20;
  std::promise<void> returned;
  std::promise<void> done;
  std::thread a([&] {
    const dyng::resources mine = res;  // the same handle, this thread's default stream
    {
      auto ws = resources_access::workspaces(mine).acquire<device_scratch>(mine);
      std::int32_t* ids = ws->ids.reserve(mine, n);
      const auto stream = static_cast<cudaStream_t>(mine.stream().get());
      EXPECT_EQ(cudaMemsetAsync(ids, 0, n * sizeof(std::int32_t), stream), cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      // Asynchronous work that is still running when the lease ends: hold the stream, then write.
      EXPECT_EQ(cudaLaunchHostFunc(stream, hold_the_stream, nullptr), cudaSuccess);
      EXPECT_EQ(cudaMemsetAsync(ids, 7, n * sizeof(std::int32_t), stream), cudaSuccess);
    }  // returned without synchronizing
    returned.set_value();
    done.get_future().wait();  // keep this thread and its default stream alive
    EXPECT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  });
  returned.get_future().wait();
  std::vector<std::int32_t> host(n, -1);
  {
    // This thread: the same workspace (the pool's only idle one), read on this thread's stream.
    auto ws = resources_access::workspaces(res).acquire<device_scratch>(res);
    std::int32_t* ids = ws->ids.reserve(res, n);  // large enough already: the same array
    EXPECT_EQ(cudaMemcpyAsync(host.data(), ids, n * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                              cudaStreamPerThread),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  }
  done.set_value();
  a.join();
  EXPECT_EQ(resources_access::workspaces(res).statistics().created, 1u);
  EXPECT_EQ(host.front(), 0x07070707);  // thread A's write, not the zeros before it
  EXPECT_EQ(host.back(), 0x07070707);
}

// Releasing idle workspaces from another thread waits for the device work of their last lease
// before their memory goes back to the pool (it is freed on the releasing thread's stream).
TEST(CudaWorkspace, ReleasingFromAnotherThreadWaitsForTheLastLease) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  std::promise<void> returned;
  std::promise<void> done;
  std::thread a([&] {
    const dyng::resources mine = res;
    {
      auto ws = resources_access::workspaces(mine).acquire<device_scratch>(mine);
      (void)ws->ids.reserve(mine, 1000);
      EXPECT_EQ(cudaLaunchHostFunc(static_cast<cudaStream_t>(mine.stream().get()), hold_the_stream,
                                   nullptr),
                cudaSuccess);
    }
    returned.set_value();
    done.get_future().wait();
  });
  returned.get_future().wait();
  const auto start = std::chrono::steady_clock::now();
  res.release_workspaces();
  const auto waited = std::chrono::steady_clock::now() - start;
  done.set_value();
  a.join();
  EXPECT_EQ(res.workspace_bytes(), 0u);
  EXPECT_GE(waited, std::chrono::milliseconds(100));  // it waited for thread A's stream
}

TEST(CudaWorkspace, HostScratchOnHostBackends) {
  const auto res = dyng::resources::sequential();
  scratch_buffer<int> s;
  EXPECT_EQ(s.capacity(), 0u);
  int* p = s.reserve(res, 10);
  ASSERT_NE(p, nullptr);
  p[9] = 3;  // host memory
  EXPECT_EQ(s.space(), dyng::memory_space::host);
  EXPECT_EQ(s.reserve(res, 5), p);
  EXPECT_EQ(s.bytes(), 10 * sizeof(int));
}

}  // namespace
