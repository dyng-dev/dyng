// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/stream.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

/// A counting host resource: checks that memory_resource_ref forwards every call.
class counting_resource {
 public:
  void* allocate(dyng::stream_ref, std::size_t bytes, std::size_t alignment) {
    ++async_allocs;
    return upstream.allocate_sync(bytes, alignment);
  }
  void deallocate(dyng::stream_ref, void* p, std::size_t bytes, std::size_t alignment) noexcept {
    ++async_frees;
    upstream.deallocate_sync(p, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    ++sync_allocs;
    return upstream.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* p, std::size_t bytes, std::size_t alignment) noexcept {
    ++sync_frees;
    upstream.deallocate_sync(p, bytes, alignment);
  }
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return dyng::memory_space::pinned_host;
  }

  int async_allocs = 0;
  int async_frees = 0;
  int sync_allocs = 0;
  int sync_frees = 0;

 private:
  dyng::host_memory_resource upstream;
};

}  // namespace

TEST(StreamRef, DefaultIsPerThreadStreamNotLegacy) {
  const dyng::stream_ref s;
  EXPECT_TRUE(s.is_per_thread_default());
  EXPECT_NE(s.get(), nullptr);  // handle 0 would be the legacy default stream
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(s.get()), 0x2U);
  const dyng::stream_ref legacy(nullptr);
  EXPECT_FALSE(legacy.is_per_thread_default());
  EXPECT_NE(s, legacy);
  EXPECT_EQ(s, dyng::stream_ref{});
  EXPECT_NO_THROW(s.synchronize());
}

TEST(HostMemoryResource, AllocatesAlignedWritableMemory) {
  dyng::host_memory_resource mr;
  EXPECT_EQ(mr.space(), dyng::memory_space::host);
  for (std::size_t alignment : {std::size_t{8}, std::size_t{64}, std::size_t{256}}) {
    void* p = mr.allocate(dyng::stream_ref{}, 1000, alignment);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignment, 0U);
    std::memset(p, 0xAB, 1000);
    mr.deallocate(dyng::stream_ref{}, p, 1000, alignment);
  }
  void* zero = mr.allocate_sync(0, 16);
  EXPECT_NE(zero, nullptr);
  mr.deallocate_sync(zero, 0, 16);
}

TEST(HostMemoryResource, RejectsBadAlignment) {
  dyng::host_memory_resource mr;
  EXPECT_THROW((void)mr.allocate_sync(64, 3), dyng::invalid_argument_error);
  EXPECT_THROW((void)mr.allocate_sync(64, 0), dyng::invalid_argument_error);
}

TEST(HostMemoryResource, HugeAllocationThrowsOutOfMemory) {
  dyng::host_memory_resource mr;
  EXPECT_THROW((void)mr.allocate_sync(std::size_t{1} << 62, 64), dyng::out_of_memory_error);
}

TEST(MemoryResourceRef, ForwardsToResource) {
  counting_resource counting;
  dyng::memory_resource_ref ref(counting);
  EXPECT_EQ(ref.space(), dyng::memory_space::pinned_host);
  void* a = ref.allocate(dyng::stream_ref{}, 128, 64);
  ref.deallocate(dyng::stream_ref{}, a, 128, 64);
  void* b = ref.allocate_sync(64, 16);
  ref.deallocate_sync(b, 64, 16);
  EXPECT_EQ(counting.async_allocs, 1);
  EXPECT_EQ(counting.async_frees, 1);
  EXPECT_EQ(counting.sync_allocs, 1);
  EXPECT_EQ(counting.sync_frees, 1);
}

TEST(MemoryResourceRef, CopiesReferToSameResource) {
  counting_resource r1;
  counting_resource r2;
  dyng::memory_resource_ref a(r1);
  dyng::memory_resource_ref b = a;  // copy, not a reference to a reference
  EXPECT_EQ(a, b);
  EXPECT_NE(a, dyng::memory_resource_ref(r2));
  void* p = b.allocate_sync(8, 8);
  b.deallocate_sync(p, 8, 8);
  EXPECT_EQ(r1.sync_allocs, 1);
}

TEST(MemoryResourceRef, DefaultHostResourceIsShared) {
  EXPECT_EQ(&dyng::default_host_memory_resource(), &dyng::default_host_memory_resource());
}
